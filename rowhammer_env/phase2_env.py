from __future__ import annotations

import base64
import binascii
import pathlib
from typing import Any

from pydantic import Field

from .mitigations import public_mitigation_capabilities
from .openenv_source import load_openenv_server_types
from .tasks import AddressResolver
from .tools.addressing import AddressError
from .worker_protocol import WorkerClient, WorkerRequest


ROOT = pathlib.Path(__file__).resolve().parents[1]
Environment, Action, Observation, State = load_openenv_server_types()

# Upper bound on how many primitive activations a single ``dram.issue`` may expand
# to. A real RowHammer flip needs O(10^4-10^5) activations (the DDR4 double-sided
# hcfirst), so compact commands must be able to request tens of thousands; this
# cap only guards the worker against a pathological/runaway compact command (the
# per-episode ``acts`` budget is the real limiter and is charged from the worker's
# true activation counter afterwards).
MAX_ISSUE_ACTIVATIONS = 2_000_000


class IssueExpansionError(Exception):
    """Raised when a ``dram.issue`` command list cannot be expanded."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _repeat_count(command: dict[str, Any], *, keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key in command:
            value = command[key]
            try:
                n = int(value)
            except (TypeError, ValueError):
                raise IssueExpansionError("BAD_SCHEMA", f"'{key}' must be an integer, got {value!r}")
            if n < 0:
                raise IssueExpansionError("BAD_SCHEMA", f"'{key}' must be non-negative, got {n}")
            return n
    return default


def expand_commands(commands: list[Any]) -> list[dict[str, Any]]:
    """Expand a ``dram.issue`` command list into primitive RD/WR/WAIT commands.

    Compact forms let a policy express a real hammer in a handful of tokens
    instead of tens of thousands of literal command objects:

    * ``{"op": "RD"|"WR"|"WAIT", ..., "repeat": N}`` (alias ``count``) issues that
      primitive ``N`` times.
    * ``{"op": "HAMMER", "rows": [addrA, addrB, ...], "pairs": N}`` (aliases
      ``addrs`` for rows, ``count`` for pairs) issues ``N`` alternating sweeps of a
      single RD to each listed row — the canonical double-sided hammer when two
      rows are given. This expands to *exactly* the explicit alternating RD
      sequence, so it is bit-for-bit equivalent at the worker (and thus in the
      disturbance model and reward) to writing every RD out by hand.

    Plain primitives without a repeat/expansion field pass through unchanged, so
    existing explicit command lists behave identically. Raises
    :class:`IssueExpansionError` on a malformed or oversized expansion.
    """
    out: list[dict[str, Any]] = []

    def _emit(primitive: dict[str, Any], times: int) -> None:
        if len(out) + times > MAX_ISSUE_ACTIVATIONS:
            raise IssueExpansionError(
                "ILLEGAL_COMMAND",
                f"dram.issue expands beyond {MAX_ISSUE_ACTIVATIONS} activations",
            )
        out.extend(primitive for _ in range(times))

    for command in commands:
        if not isinstance(command, dict):
            raise IssueExpansionError("BAD_SCHEMA", "each command must be an object")
        op = command.get("op")
        if op == "HAMMER":
            rows = command.get("rows", command.get("addrs"))
            if not isinstance(rows, list) or not rows:
                raise IssueExpansionError("BAD_SCHEMA", "HAMMER requires a non-empty 'rows' list")
            pairs = _repeat_count(command, keys=("pairs", "count", "repeat"), default=0)
            for _ in range(pairs):
                for addr in rows:
                    _emit({"op": "RD", "addr": addr}, 1)
        elif op in ("RD", "WR", "WAIT"):
            times = _repeat_count(command, keys=("repeat", "count"), default=1)
            _emit(command, times)
        else:
            raise IssueExpansionError("ILLEGAL_COMMAND", str(op))
    return out


class Phase2Action(Action):
    # ``tool`` defaults to empty so a malformed action (missing tool) parses at the
    # transport layer and is rejected by ``step`` with the SPEC §8 ``BAD_SCHEMA``
    # code, instead of surfacing OpenEnv's transport-level ``VALIDATION_ERROR``
    # (which is not in the stable error set). The normative structural contract
    # still requires ``tool`` (spec/schemas/action.schema.json).
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class Phase2Observation(Observation):
    cycle: int = 0
    data_b64: str | None = None
    last_action: dict[str, Any] = Field(default_factory=dict)
    public_counters: dict[str, Any] = Field(default_factory=dict)
    feedback: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, str] | None = None
    # Policy-facing mirror of ``metadata``. OpenEnv's ``serialize_observation``
    # drops ``metadata`` from the wire payload (it treats it as server-internal),
    # which would strip the SPEC §8 initial/step observation fields (objective,
    # target, disclosure, budget_remaining) for any policy attached over HTTP.
    # ``model_dump`` folds ``metadata`` into this serialized field so they survive.
    info: dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        if "info" in data:
            data["info"] = {**self.metadata, **(self.info or {})}
        return data


class Phase2State(State):
    cycle: int = 0


class RowHammerEnv(Environment[Phase2Action, Phase2Observation, Phase2State]):
    def __init__(
        self,
        worker_path: pathlib.Path | None = None,
        config_path: pathlib.Path | None = None,
    ) -> None:
        super().__init__()
        self.worker_path = worker_path or ROOT / "build/phase2/ramulator_worker"
        # Phase 2+ uses the plugin-enabled config so the worker emits the real
        # issued-command stream (P11). Phase 1's smoke binary keeps the plain
        # phase-1 config, which has no IssuedEventRecorder registered.
        self.config_path = config_path or ROOT / "build/phase2/p2_external_ddr4.yaml"
        self._base_config_path = self.config_path
        self._state = Phase2State(episode_id=None, step_count=0, cycle=0)
        self._worker: WorkerClient | None = None
        # Address projection + disclosure enforcement (P12). Absent in the bare
        # phase-2 env (no geometry yet), which then only accepts logical addresses.
        self._resolver: AddressResolver | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **_: Any,
    ) -> Phase2Observation:
        self.close()
        self._state = Phase2State(episode_id=episode_id or "p2_episode", step_count=0, cycle=0)
        if not self.worker_path.is_file() or not self.config_path.is_file():
            self._worker = None
            return self._error("UNAVAILABLE_CAPABILITY", "Phase 2 worker/config is not built")
        try:
            self._worker = WorkerClient(self.worker_path, self.config_path)
        except FileNotFoundError:
            self._worker = None
            return self._error("UNAVAILABLE_CAPABILITY", "Phase 2 worker/config is not built")
        return Phase2Observation(
            reward=0.0,
            done=False,
            metadata={
                "seed": seed,
                "allowed_tools": ["dram.info", "dram.read", "dram.write", "dram.issue", "episode.finish"],
                "commands": ["RD", "WR", "WAIT"],
                "mitigations": public_mitigation_capabilities(),
            },
        )

    def step(self, action: Phase2Action, timeout_s: float | None = None, **_: Any) -> Phase2Observation:
        del timeout_s
        if not action.tool:
            # Malformed action (no tool): reject with a stable SPEC §8 code and no
            # state mutation, rather than OpenEnv's transport-level VALIDATION_ERROR.
            return self._error("BAD_SCHEMA", "action.tool is required")
        self._state.step_count += 1
        try:
            if action.tool == "dram.info":
                forms = sorted(self._resolver.disclosure.allowed_forms()) if self._resolver else ["logical"]
                return Phase2Observation(
                    reward=0.0,
                    done=False,
                    cycle=self._state.cycle,
                    metadata={
                        "address_forms": forms,
                        "commands": ["RD", "WR", "WAIT"],
                        "mitigations": public_mitigation_capabilities(),
                    },
                )
            if action.tool == "episode.finish":
                self.close()
                return Phase2Observation(reward=0.0, done=True, cycle=self._state.cycle)
            if self._worker is None:
                return self._error("UNAVAILABLE_CAPABILITY", "episode is not reset or worker is unavailable")

            if action.tool == "dram.read":
                req = WorkerRequest("READ", self._next_id(), (str(self._logical_addr(action.args)), str(action.args.get("length", 64))))
                return self._from_worker(self._worker.call(req))
            if action.tool == "dram.write":
                raw = base64.b64decode(str(action.args.get("data_b64", "")), validate=True)
                req = WorkerRequest("WRITE", self._next_id(), (str(self._logical_addr(action.args)), raw.hex()))
                return self._from_worker(self._worker.call(req))
            if action.tool == "dram.issue":
                return self._issue(action.args)
            return self._error("UNSUPPORTED_TOOL", action.tool)
        except AddressError as exc:
            return self._error(exc.code, exc.message)
        except (KeyError, ValueError, TypeError, binascii.Error) as exc:
            return self._error("BAD_SCHEMA", str(exc))

    @property
    def state(self) -> Phase2State:
        return self._state

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def _issue(self, args: dict[str, Any]) -> Phase2Observation:
        commands = args.get("commands")
        if not isinstance(commands, list) or not commands:
            return self._error("BAD_SCHEMA", "dram.issue requires commands")
        try:
            # Expand compact forms (repeat/HAMMER) into primitive RD/WR/WAIT
            # commands. Each primitive still runs through the real worker below, so
            # the activation count charged against the budget and seen by the
            # disturbance model is the true expanded count — nothing is fabricated.
            primitives = expand_commands(commands)
        except IssueExpansionError as exc:
            return self._error(exc.code, exc.message)
        last = Phase2Observation(reward=0.0, done=False, cycle=self._state.cycle)
        for command in primitives:
            op = command["op"]
            if op == "WAIT":
                req = WorkerRequest("ISSUE", self._next_id(), ("WAIT", str(command.get("cycles", 0))))
            elif op == "RD":
                req = WorkerRequest("ISSUE", self._next_id(), ("RD", str(self._addr_value(command))))
            else:  # WR (expand_commands only yields RD/WR/WAIT)
                raw = base64.b64decode(str(command.get("data_b64", "")), validate=True)
                req = WorkerRequest("ISSUE", self._next_id(), ("WR", str(self._addr_value(command)), raw.hex()))
            last = self._from_worker(self._worker.call(req))  # type: ignore[union-attr]
            if last.error:
                return last
        return last

    def _logical_addr(self, args: dict[str, Any]) -> int:
        return self._addr_value(args)

    def _addr_value(self, container: dict[str, Any]) -> int:
        addr = container.get("addr", container)
        if self._resolver is not None:
            return self._resolver.to_linear(addr)
        # No disclosure/geometry wired yet: only logical addressing is available,
        # and any other form fails closed rather than being silently accepted.
        # A bare int is shorthand for {"kind":"logical","addr":N}, same as
        # AddressResolver.to_linear.
        if isinstance(addr, int) and not isinstance(addr, bool):
            addr = {"kind": "logical", "addr": addr}
        if not isinstance(addr, dict):
            raise AddressError("BAD_SCHEMA", "address must be an object")
        if addr.get("kind") != "logical":
            raise AddressError("ADDRESS_NOT_DISCLOSED", "only logical addresses are disclosed")
        try:
            return int(addr["addr"])
        except (KeyError, TypeError, ValueError):
            raise AddressError("BAD_SCHEMA", "logical address requires an integer 'addr'")

    def _next_id(self) -> str:
        return f"a{self._state.step_count}"

    def _from_worker(self, payload: dict[str, Any]) -> Phase2Observation:
        if not payload.get("ok"):
            err = payload.get("error") or {}
            return self._error(err.get("code", "INTERNAL_SIMULATOR_ERROR"), err.get("message", "worker failed"))
        self._state.cycle = int(payload["cycle"])
        data_hex = payload.get("data_hex")
        data_b64 = base64.b64encode(bytes.fromhex(data_hex)).decode() if data_hex else None
        return Phase2Observation(
            reward=0.0,
            done=False,
            cycle=self._state.cycle,
            data_b64=data_b64,
            last_action=payload.get("last_action", {}),
            public_counters=payload.get("public_counters", {}),
            feedback={"trace_tail": payload.get("events", [])},
        )

    def _error(self, code: str, message: str) -> Phase2Observation:
        return Phase2Observation(reward=0.0, done=True, cycle=self._state.cycle, error={"code": code, "message": message})
