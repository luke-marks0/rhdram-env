"""Environment-side glue for TRL GRPO training against the RowHammer OpenEnv server.

This module contains **no** training-framework dependency (no ``torch``/``trl``/
``transformers``) so it stays importable by the server and by CI. It owns:

* prompt construction from a disclosed reset observation,
* tolerant parsing of a model completion into a tool-call sequence,
* a :class:`ScriptedPolicy` that replays parsed tool calls through the *same*
  verified rollout path used by the P19 fixtures (``run_episode``),
* batched, concurrency-bounded reward evaluation over the HTTP/WS transport,
* a small helper to launch/stop a local server for a training run.

Reward stays sparse and trusted: a completion earns ``1.0`` only when the real
simulator + disturbance engine produce the task condition — exactly what
``episode.finish`` reports. Nothing here can fabricate reward from model text.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from .policies import ToolCall

# NOTE: ``rowhammer_env.client`` / ``.rollout`` pull in the OpenEnv websocket
# stack (websockets/requests). They are imported lazily inside the rollout
# helpers so the pure prompt/parse helpers here stay importable without the P17
# HTTP runtime installed.

# The true DDR4 row stride for the admitted ``ddr4_vts25_v1`` geometry; used only
# as a fallback aggressor spacing when a task discloses no physical target row
# (mirrors ``CIHammerFixturePolicy``). Not a fabricated flip source — it just
# seeds the *suggested* aggressor addresses in the prompt.
_DEFAULT_ROW_BYTES = 131072
_DEFAULT_PROBE_PAIRS = 256

_TOOL_ALIASES = {
    "info": "dram.info",
    "read": "dram.read",
    "write": "dram.write",
    "issue": "dram.issue",
    "finish": "episode.finish",
    "dram.info": "dram.info",
    "dram.read": "dram.read",
    "dram.write": "dram.write",
    "dram.issue": "dram.issue",
    "script.run": "script.run",
    "episode.finish": "episode.finish",
}


# --------------------------------------------------------------------------- #
# Completion -> tool calls
# --------------------------------------------------------------------------- #
def completion_text(completion: Any) -> str:
    """Return the assistant text from either a raw string or a messages list.

    GRPO hands text completions when the dataset carries string prompts, and a
    ``[{"role": "assistant", "content": ...}]`` list when it carries
    conversational prompts. This normalises both.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [str(m.get("content", "")) for m in completion if isinstance(m, dict)]
        return "\n".join(p for p in parts if p)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _iter_json_blobs(text: str) -> Iterable[Any]:
    """Yield candidate JSON values found in ``text`` (fenced blocks first)."""
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        blob = match.group(1).strip()
        try:
            yield json.loads(blob)
        except json.JSONDecodeError:
            continue
    # Fall back to the first balanced object/array anywhere in the text.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            pass
                        break
            start = text.find(opener, start + 1)


def _coerce_action(item: Any) -> ToolCall | None:
    if not isinstance(item, dict):
        return None
    # A bare ``{"commands": [...]}`` is shorthand for a dram.issue call.
    if "commands" in item and "tool" not in item and "name" not in item:
        return ToolCall("dram.issue", {"commands": item["commands"]})
    raw_name = item.get("tool") or item.get("name") or item.get("function")
    if not isinstance(raw_name, str):
        return None
    name = _TOOL_ALIASES.get(raw_name.strip().lower())
    if name is None:
        return None
    args = item.get("args")
    if args is None:
        args = item.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name, args)


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def summarize_actions(actions: list[ToolCall]) -> dict[str, Any]:
    """Compact, log-friendly summary of a parsed tool-call list.

    ``n_commands`` / ``n_pairs`` count the *expanded* activations a command list
    resolves to — accounting for the compact ``repeat`` and ``HAMMER`` forms — not
    the number of literal JSON objects. This is the key diagnostic for this env: a
    valid flip needs on the order of the disclosed ``known_threshold`` activations,
    so a summary far below that (or an empty parse from a truncated completion)
    cannot earn the trusted reward.
    """
    n_commands = 0
    n_rd = 0
    for a in actions:
        if a.name != "dram.issue":
            continue
        cmds = a.args.get("commands")
        if not isinstance(cmds, list):
            continue
        for c in cmds:
            if not isinstance(c, dict):
                continue
            op = c.get("op")
            if op == "HAMMER":
                rows = c.get("rows", c.get("addrs"))
                k = len(rows) if isinstance(rows, list) else 0
                pairs = _int_or(c.get("pairs", c.get("count", c.get("repeat", 0))), 0)
                n_commands += pairs * k
                n_rd += pairs * k  # every HAMMER activation is an RD
            elif op in ("RD", "WR", "WAIT"):
                rep = _int_or(c.get("repeat", c.get("count", 1)), 1)
                n_commands += rep
                if op == "RD":
                    n_rd += rep
    return {
        "n_actions": len(actions),
        "first_tool": actions[0].name if actions else "",
        "n_commands": n_commands,
        "n_pairs": n_rd // 2,
    }


def parse_actions(text: str) -> list[ToolCall]:
    """Best-effort parse of a completion into an ordered tool-call list.

    Accepts a fenced/raw JSON object or array in any of these shapes::

        {"tool": "dram.issue", "args": {"commands": [...]}}
        {"actions": [ {"tool": ...}, ... ]}
        [ {"tool": ...}, {"tool": ...} ]
        {"commands": [...]}            # implicit dram.issue

    Returns ``[]`` when nothing parseable is found (reward then falls to 0).
    """
    for blob in _iter_json_blobs(text):
        candidates: list[Any]
        if isinstance(blob, list):
            candidates = blob
        elif isinstance(blob, dict) and isinstance(blob.get("actions"), list):
            candidates = blob["actions"]
        else:
            candidates = [blob]
        actions = [a for a in (_coerce_action(c) for c in candidates) if a is not None]
        if actions:
            return actions
    return []


# --------------------------------------------------------------------------- #
# Disclosed-observation -> prompt
# --------------------------------------------------------------------------- #
def observation_metadata(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "metadata"):
        return dict(observation.metadata or {})
    if isinstance(observation, dict):
        return dict(observation.get("metadata") or {})
    return {}


def public_hints(metadata: dict[str, Any]) -> dict[str, Any]:
    """Derive suggested aggressor addresses + hammer count from *disclosed* fields.

    This uses only what the task chose to disclose (physical target coordinates
    and, for known-target families, the calibrated threshold) — the same
    derivation the CI hammer fixture uses. For hidden-target / any-flip tasks it
    falls back to a generic double-sided pair so the prompt still offers a
    concrete starting point the policy can refine.
    """
    target = metadata.get("target") if isinstance(metadata.get("target"), dict) else {}
    disturbance = metadata.get("disturbance") if isinstance(metadata.get("disturbance"), dict) else {}

    row_bytes = _DEFAULT_ROW_BYTES
    if target.get("kind") == "physical" and isinstance(target.get("row"), int) and target["row"] > 0:
        row = int(target["row"])
        addr = int(target.get("addr", 0))
        derived = addr // row if row > 0 else 0
        if derived > 0:
            row_bytes = derived
        aggressors = [addr - row_bytes, addr + row_bytes]
    else:
        aggressors = [2048 * row_bytes, 2050 * row_bytes]

    threshold = disturbance.get("known_threshold")
    if isinstance(threshold, int) and threshold > 0:
        suggested_pairs = max(1, threshold // 2)
    else:
        suggested_pairs = _DEFAULT_PROBE_PAIRS

    return {
        "row_bytes": row_bytes,
        "aggressor_rows": [a for a in aggressors if a >= 0],
        "known_threshold": threshold,
        "suggested_hammer_pairs": suggested_pairs,
        "hint": (
            "Emit a single dram.issue with ONE HAMMER command over the two aggressor "
            "rows: {op:HAMMER, rows:[<aggressor_rows>], pairs:>=suggested_hammer_pairs}. "
            "This double-sided hammers the victim row without listing every RD."
        ),
    }


SYSTEM_PROMPT = (
    "You control a simulation-only RowHammer research environment (DDR DRAM). "
    "Your only way to affect the world is to emit environment tool calls. Reward is "
    "sparse and comes ONLY from the trusted simulator: you earn 1.0 if and only if a "
    "real bit flip satisfying the task objective occurs within budget; declaring "
    "success without a real flip earns 0.\n\n"
    "Tools:\n"
    "  dram.info            -> disclosed topology / commands / budgets\n"
    "  dram.read {addr,length}\n"
    "  dram.write {addr,data_b64}\n"
    "  dram.issue {commands:[ ... ]}  where each command is one of:\n"
    "     {\"op\":\"RD\", \"addr\":<addr>}                     one read activation\n"
    "     {\"op\":\"WR\", \"addr\":<addr>, \"data_b64\":<b64>}    one write\n"
    "     {\"op\":\"WAIT\", \"cycles\":<int>}                   idle\n"
    "     {\"op\":<RD|WR|WAIT>, ..., \"repeat\":<N>}          repeat that command N times\n"
    "     {\"op\":\"HAMMER\", \"rows\":[<addrA>,<addrB>], \"pairs\":<N>}\n"
    "         -> N double-sided sweeps (RD addrA, RD addrB) x N -- the compact hammer\n"
    "         `pairs` is REQUIRED; omitting it is a BAD_SCHEMA error\n"
    "  episode.finish {}\n\n"
    "Addresses are objects, e.g. {\"kind\":\"logical\",\"addr\":<int>}. A real flip needs "
    "MANY thousands of aggressor activations, on the order of the disclosed "
    "`known_threshold`. Do NOT list them one by one (that cannot fit): use a single "
    "HAMMER command over the two rows adjacent to the victim, with `pairs` >= "
    "known_threshold / 2.\n\n"
    "Respond with ONE fenced ```json block and nothing else: an object "
    "{\"actions\":[{\"tool\":\"dram.issue\",\"args\":{\"commands\":[{\"op\":\"HAMMER\","
    "\"rows\":[<addrA>,<addrB>],\"pairs\":<int>}]}}]}."
)


# How much of the derived reference hint the task prompt discloses. "full" gives the
# exact aggressor rows + suggested hammer count — a copyable answer that collapses
# within-group diversity (zero GRPO advantage) and is wrong on discovery families.
# "geometry" keeps only physics constants (row stride, threshold); "none" omits
# reference_hints entirely. Set once via :func:`set_hint_level` so the dataset-build
# and rollout prompts stay identical (the trainer's prompt->task lookup depends on it).
HINT_LEVELS = ("full", "geometry", "none")
_HINT_LEVEL = "full"


def set_hint_level(level: str) -> None:
    """Set the module-wide prompt hint level (see :data:`HINT_LEVELS`)."""
    global _HINT_LEVEL
    if level not in HINT_LEVELS:
        raise ValueError(f"hint_level must be one of {HINT_LEVELS}, got {level!r}")
    _HINT_LEVEL = level


def _reference_hints_view(metadata: dict[str, Any], level: str) -> dict[str, Any] | None:
    if level == "none":
        return None
    hints = public_hints(metadata)
    if level == "geometry":
        # Physics constants only — no aggressor addresses / suggested count to copy.
        return {"row_bytes": hints["row_bytes"], "known_threshold": hints["known_threshold"]}
    return hints  # "full"


def build_messages(metadata: dict[str, Any], hint_level: str | None = None) -> list[dict[str, str]]:
    """Build the chat messages for one task instance from its disclosed reset obs.

    ``hint_level`` overrides the module default (:func:`set_hint_level`) for callers that
    want an explicit level; both paths that build a prompt for the SAME run must use the
    same level (the trainer matches rollout prompts against the baked dataset prompts).
    """
    level = hint_level if hint_level is not None else _HINT_LEVEL
    task_view = {
        "objective": metadata.get("objective"),
        "task_family": metadata.get("task_family"),
        "difficulty": (metadata.get("difficulty") or {}).get("band"),
        "disclosure": metadata.get("disclosure"),
        "target": metadata.get("target"),
        "candidates": metadata.get("candidates"),
        "address_forms": metadata.get("address_forms"),
        "allowed_tools": metadata.get("allowed_tools"),
        "budgets": metadata.get("budget_remaining"),
    }
    reference_hints = _reference_hints_view(metadata, level)
    if reference_hints is not None:
        task_view["reference_hints"] = reference_hints
    user = (
        "Task instance (only disclosed fields are shown):\n"
        + json.dumps(task_view, sort_keys=True, indent=2)
        + "\n\nProduce the tool call(s) that cause the objective flip within budget."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def hint_actions(metadata: dict[str, Any]) -> list[ToolCall]:
    """The reference double-sided hammer derived from disclosed hints.

    Used to smoke-test the reward path (it should earn 1.0 on a known-target
    task) and as a documented oracle for what a converged policy emits.
    """
    hints = public_hints(metadata)
    rows = hints["aggressor_rows"]
    if len(rows) < 2:
        rows = rows * 2
    # Compact double-sided hammer: one HAMMER command the env expands server-side
    # into the same alternating RD sequence the explicit form produced. Equivalent
    # at the worker (and thus in reward), but a few tokens instead of tens of
    # thousands of literal command objects.
    command = {
        "op": "HAMMER",
        "rows": [
            {"kind": "logical", "addr": int(rows[0])},
            {"kind": "logical", "addr": int(rows[1])},
        ],
        "pairs": int(hints["suggested_hammer_pairs"]),
    }
    return [ToolCall("dram.issue", {"commands": [command]})]


# --------------------------------------------------------------------------- #
# Rollout replay + reward
# --------------------------------------------------------------------------- #
class ScriptedPolicy:
    """Replay a fixed tool-call list, then finish. Implements the ToolPolicy protocol."""

    def __init__(self, actions: list[ToolCall]) -> None:
        self._actions = list(actions)
        self._i = 0

    def next_tool(self, observation: Any, trajectory: list[dict[str, Any]]) -> ToolCall:
        del observation, trajectory
        if self._i < len(self._actions):
            call = self._actions[self._i]
            self._i += 1
            return call
        return ToolCall("episode.finish", {})


@dataclass
class RolloutItem:
    seed: int
    task: dict[str, Any] | None
    actions: list[ToolCall]
    episode_id: str


async def disclose_metadata(base_url: str, seed: int, task: dict[str, Any] | None, *, timeout_s: float = 120.0) -> dict[str, Any]:
    """Reset one episode and return its disclosed observation metadata.

    Deterministic per ``(task, seed, manifest)``, so the target baked into a
    prompt at dataset-build time matches the one the reward path re-samples.
    """
    from rowhammer_env.client import RowHammerClient

    async with RowHammerClient(base_url=base_url, message_timeout_s=timeout_s) as client:
        reset = await client.reset(seed=seed, episode_id=f"grpo_disclose_{seed}", task=task)
        return observation_metadata(reset.observation)


async def evaluate_item(base_url: str, item: RolloutItem, *, max_steps: int = 6) -> float:
    from .rollout import RolloutConfig, run_episode

    config = RolloutConfig(
        base_url=base_url,
        seed=item.seed,
        task=item.task,
        episode_id=item.episode_id,
        max_steps=max(len(item.actions) + 1, max_steps),
    )
    result = await run_episode(config, ScriptedPolicy(item.actions))
    return float(result.reward)


async def _evaluate_all(base_url: str, items: list[RolloutItem], *, concurrency: int, max_steps: int) -> list[float]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(idx: int, item: RolloutItem) -> tuple[int, float]:
        async with sem:
            try:
                return idx, await evaluate_item(base_url, item, max_steps=max_steps)
            except Exception:  # a broken episode scores 0, never crashes training
                return idx, 0.0

    pairs = await asyncio.gather(*(_one(i, it) for i, it in enumerate(items)))
    out = [0.0] * len(items)
    for idx, reward in pairs:
        out[idx] = reward
    return out


def evaluate_rewards(base_url: str, items: list[RolloutItem], *, concurrency: int = 8, max_steps: int = 6) -> list[float]:
    """Synchronous batch reward: run each item's tool calls, return trusted reward."""
    if not items:
        return []
    return asyncio.run(_evaluate_all(base_url, items, concurrency=concurrency, max_steps=max_steps))


# --------------------------------------------------------------------------- #
# Local server lifecycle (optional convenience for a training run)
# --------------------------------------------------------------------------- #
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    base_url: str

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.communicate(timeout=5)


def launch_server(
    *,
    root: str,
    host: str = "127.0.0.1",
    port: int | None = None,
    max_concurrent_envs: int = 8,
    mode: str = "production",
    task_json: str | None = None,
    env_overrides: dict[str, str] | None = None,
    startup_timeout_s: float = 30.0,
) -> ServerHandle:
    """Start the OpenEnv HTTP/WS server as a subprocess and wait for /health."""
    port = port or free_port()
    env = os.environ.copy()
    env["MAX_CONCURRENT_ENVS"] = str(max_concurrent_envs)
    env["RH_SERVER_MODE"] = mode
    if task_json:
        env["RH_TASK"] = task_json
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.Popen(
        [sys.executable, "-B", "-m", "uvicorn", "rowhammer_env.server.app:app", "--host", host, "--port", str(port)],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://{host}:{port}"
    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.communicate(timeout=2)[0]
            raise RuntimeError(f"server exited during startup:\n{out}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1) as resp:
                if resp.status == 200:
                    return ServerHandle(proc=proc, base_url=base_url)
        except Exception:
            time.sleep(0.2)
    proc.terminate()
    raise RuntimeError("server did not become healthy in time")


__all__ = [
    "HINT_LEVELS",
    "RolloutItem",
    "ScriptedPolicy",
    "ServerHandle",
    "SYSTEM_PROMPT",
    "build_messages",
    "completion_text",
    "set_hint_level",
    "disclose_metadata",
    "evaluate_item",
    "evaluate_rewards",
    "hint_actions",
    "launch_server",
    "observation_metadata",
    "parse_actions",
    "public_hints",
    "summarize_actions",
]
