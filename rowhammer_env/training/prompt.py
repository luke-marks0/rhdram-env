"""Turn the environment's structured observations into chat turns, and parse the
model's fenced-JSON tool call back into an action.

The policy emits exactly one tool call per turn as a ```json fenced block:

    ```json
    {"tool": "dram.issue", "args": {"commands": [{"op": "RD", "addr": {"kind": "logical", "addr": 123}}]}}
    ```

Kept deliberately small: the PoC surface is only ``dram.info``, ``dram.issue``,
``episode.finish``. This module has no torch dependency so the rollout driver and
tests can render/parse without the model stack.
"""
from __future__ import annotations

import json
import re
from typing import Any

# The tool role Qwen3's chat template understands; masking keys off role ==
# "assistant", so tool results are never trained on regardless.
TOOL_ROLE = "tool"

# full_trace repeats dozens of near-identical event rows every turn, bloating the
# prompt with signal the digest (acts_delta / per_addr_hits) already carries. Keep
# only the last few rows; 0 drops the trace entirely.
DEFAULT_TRACE_TAIL = 4

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE = re.compile(r"(\{(?:[^{}]|\{[^{}]*\})*\})", re.DOTALL)

SYSTEM_PROMPT = """\
You are an agent probing a simulated DDR4 memory for RowHammer bit flips. The \
simulator is cycle-accurate; nothing you do touches real hardware. Your goal is to \
cause a trusted bit flip in the victim row within the activation and tool-call budget.

You act by emitting exactly ONE tool call per turn, as a single fenced JSON block:

```json
{"tool": "<name>", "args": {...}}
```

Available tools:
- dram.info  -> {"tool": "dram.info", "args": {}}
    Returns geometry, commands, admitted mitigations, and remaining budget.
- dram.issue -> {"tool": "dram.issue", "args": {"commands": [ ... ]}}
    commands is a non-empty list. Primitives: {"op": "RD"|"WR"|"WAIT", "addr": ADDR}
    (WAIT takes {"op":"WAIT","cycles":N}). Compact forms:
      {"op": "RD", "addr": ADDR, "repeat": N}
      {"op": "HAMMER", "rows": [ADDR, ADDR], "pairs": N}   # N sweeps of one RD to each row
    ADDR is {"kind": "logical", "addr": <int>} or {"kind": "handle", "id": "<id>"}.
- episode.finish -> {"tool": "episode.finish", "args": {}}
    Ends the episode. Only do this once you believe the flip has happened.

How to win: only the two rows physically adjacent to the victim IN THE SAME BANK can \
flip it (a double-sided hammer). The address-to-bank mapping is hidden and NOT the \
numeric order of addresses, so you must discover which candidates share the victim's \
bank using the timing side channel: alternately reading two rows in the SAME bank \
forces new row activations (the returned timing_digest shows acts_delta > 0), while \
two rows in DIFFERENT banks stay open (acts_delta == 0). Probe candidates against the \
victim, keep the same-bank ones, then HAMMER them enough times to cross the flip \
threshold. Think briefly, then emit one tool call."""


def system_message() -> dict[str, str]:
    return {"role": "system", "content": SYSTEM_PROMPT}


def initial_user_message(obs: Any) -> dict[str, str]:
    """The opening turn: objective, disclosed target/candidates, budget, threshold."""
    m = obs.metadata
    lines = ["New episode.", ""]
    objective = m.get("objective", {})
    lines.append(f"Objective: {json.dumps(objective)}")
    if "candidates" in m:
        lines.append(f"Candidate rows ({len(m['candidates'])}): {json.dumps(m['candidates'])}")
    dist = m.get("disturbance", {})
    if "known_threshold" in dist:
        lines.append(f"Known flip threshold (hammer sweeps needed): {dist['known_threshold']}")
    lines.append(f"Address forms accepted: {m.get('address_forms')}")
    lines.append(f"Budget remaining: {json.dumps(m.get('budget_remaining'))}")
    lines.append("")
    lines.append("Emit your first tool call.")
    return {"role": "user", "content": "\n".join(lines)}


def tool_result_message(obs: Any, *, trace_tail: int = DEFAULT_TRACE_TAIL) -> dict[str, str]:
    """Render one env step result as the tool-role turn fed back to the policy."""
    payload: dict[str, Any] = {
        "reward": obs.reward,
        "done": obs.done,
        "public_counters": obs.public_counters,
        "budget_remaining": obs.metadata.get("budget_remaining"),
    }
    if obs.error:
        payload["error"] = obs.error
    feedback = _trim_feedback(obs.feedback, trace_tail)
    if feedback:
        payload["feedback"] = feedback
    # dram.info answers into metadata; surface the parts a policy needs.
    for key in ("geometry", "commands"):
        if key in obs.metadata:
            payload[key] = obs.metadata[key]
    return {"role": TOOL_ROLE, "content": json.dumps(payload)}


def _trim_feedback(feedback: Any, limit: int) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        return {}
    out = dict(feedback)
    tail = out.get("trace_tail")
    if isinstance(tail, list) and len(tail) > limit:
        out["trace_tail_len"] = len(tail)
        out["trace_tail"] = tail[-limit:] if limit > 0 else []
    return out


def render_action(action: dict[str, Any]) -> str:
    """Render an action dict as the fenced-JSON completion the prompt asks for.

    Round-trips through :func:`parse_action`, so a scripted policy rendered this way
    is trace-equivalent to a model that emitted the same block.
    """
    body = json.dumps({"tool": action["tool"], "args": action.get("args", {})})
    return "```json\n" + body + "\n```"


def parse_action(text: str) -> dict[str, Any]:
    """Extract the tool call from a model completion.

    Returns ``{"tool": str, "args": dict}``. A completion with no parseable block
    yields ``{"tool": "", "args": {}}`` — the env rejects an empty tool with
    ``BAD_SCHEMA`` and the episode keeps running, so a malformed turn costs a tool
    call but does not crash the rollout. When several blocks appear (a model that
    "thinks" in JSON), the last one wins.
    """
    candidates = _FENCE.findall(text) or _BARE.findall(text)
    for blob in reversed(candidates):
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            args = obj.get("args", {})
            return {"tool": obj["tool"], "args": args if isinstance(args, dict) else {}}
    return {"tool": "", "args": {}}
