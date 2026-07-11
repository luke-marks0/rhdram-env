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

# Upper bound on the number of *raw* issued events echoed back in
# ``feedback.trace_tail`` from a single ``dram.issue``. ``_issue`` runs the worker
# once per expanded primitive and each drains its own events, so a multi-primitive
# probe or a tens-of-thousands-activation ``HAMMER`` would otherwise either lose
# every intermediate primitive's events (old behaviour) or flood the prompt. We
# aggregate across primitives and keep only the last ``ISSUE_TRACE_TAIL_CAP``
# events verbatim; the full timing summary survives — losslessly for the counts
# that matter — in the ``timing_digest`` (P22).
ISSUE_TRACE_TAIL_CAP = 64


class _TimingDigest:
    """Bounded, coordinate-free summary of the true events of one ``dram.issue``.

    Aggregated across the expanded primitives (0.2.2: ``_issue`` calls the worker
    once per primitive and each ``drain()``s its own events, so a single returned
    observation otherwise carries only the *last* primitive's timing). Every field
    is derived from real issued events — nothing fabricated — and carries only
    counts/clocks, never a decoded coordinate, so it is safe to expose under
    ``logical_only`` mapping by construction. ``per_addr_hits`` is keyed by the
    address token the policy itself supplied (a handle id or a logical address it
    already knows), never a resolved-hidden linear address, so it leaks nothing a
    handle/logical task did not already disclose.

    The bank-conflict discriminator (0.2.1) is ``acts_delta`` (equivalently the
    per-address ``acts`` count): a policy ``RD``'s own ``row_hit`` is always true,
    so what distinguishes same-bank from different-bank is whether an alternating
    access *forced a new ACT*, which shows up here as ACT events, not as a false
    ``row_hit``.
    """

    __slots__ = ("acts_delta", "cycles_delta", "first_clk", "last_clk", "per_addr")

    def __init__(self) -> None:
        self.acts_delta = 0
        self.cycles_delta = 0
        self.first_clk: int | None = None
        self.last_clk: int | None = None
        self.per_addr: dict[str, dict[str, int]] = {}

    def absorb(self, key: str | None, last_action: dict[str, Any], events: list[dict[str, Any]]) -> None:
        self.cycles_delta += int(last_action.get("cycle_delta", 0) or 0)
        bucket = self.per_addr.setdefault(key, {"acts": 0, "hits": 0, "misses": 0}) if key is not None else None
        for event in events:
            clk = event.get("clk")
            if clk is not None:
                if self.first_clk is None:
                    self.first_clk = int(clk)
                self.last_clk = int(clk)
            if event.get("op") == "ACT":
                self.acts_delta += 1
                if bucket is not None:
                    bucket["acts"] += 1
            if bucket is not None:
                if event.get("row_hit"):
                    bucket["hits"] += 1
                else:
                    bucket["misses"] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "acts_delta": self.acts_delta,
            "cycles_delta": self.cycles_delta,
            "first_clk": self.first_clk if self.first_clk is not None else 0,
            "last_clk": self.last_clk if self.last_clk is not None else 0,
            "per_addr_hits": self.per_addr,
        }


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
                metadata: dict[str, Any] = {
                    "address_forms": forms,
                    "commands": ["RD", "WR", "WAIT"],
                    "mitigations": public_mitigation_capabilities(),
                }
                geometry = self._geometry_block()
                if geometry is not None:
                    metadata["geometry"] = geometry
                return Phase2Observation(reward=0.0, done=False, cycle=self._state.cycle, metadata=metadata)
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

    def _geometry_block(self) -> dict[str, Any] | None:
        """Public standard geometry for ``dram.info`` (P21); ``None`` when unknown.

        The bare phase-2 env fetches no worker geometry, so it has none to disclose;
        the disturbance/task envs override this once the worker has reported its
        level sizes.
        """
        return None

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
        digest = _TimingDigest()
        trace_tail: list[dict[str, Any]] = []
        flips = 0
        oracle_refreshes = 0
        public_flips: list[dict[str, Any]] = []
        last = Phase2Observation(reward=0.0, done=False, cycle=self._state.cycle)
        for command in primitives:
            op = command["op"]
            if op == "WAIT":
                key = None
                req = WorkerRequest("ISSUE", self._next_id(), ("WAIT", str(command.get("cycles", 0))))
            elif op == "RD":
                key = self._digest_addr_key(command)
                req = WorkerRequest("ISSUE", self._next_id(), ("RD", str(self._addr_value(command))))
            else:  # WR (expand_commands only yields RD/WR/WAIT)
                key = self._digest_addr_key(command)
                raw = base64.b64decode(str(command.get("data_b64", "")), validate=True)
                req = WorkerRequest("ISSUE", self._next_id(), ("WR", str(self._addr_value(command)), raw.hex()))
            last = self._from_worker(self._worker.call(req))  # type: ignore[union-attr]
            if last.error:
                return last
            # ``last.feedback`` is already projected to the disclosure level by the
            # per-primitive ``_from_worker`` override, so aggregating it here keeps
            # the leakage guard: coordinates are already stripped from these events.
            events = last.feedback.get("trace_tail", [])
            digest.absorb(key, last.last_action, events)
            trace_tail.extend(events)
            flips += int(last.feedback.get("new_public_flips", 0) or 0)
            oracle_refreshes += int(last.feedback.get("oracle_refreshes", 0) or 0)
            public_flips.extend(last.feedback.get("public_flips", []) or [])
        # Aggregate the whole issue's feedback onto the last primitive's observation
        # (which carries the final cumulative ``public_counters``/``cycle``) instead
        # of surfacing only the last primitive's slice (0.2.2).
        if "new_public_flips" in last.feedback:
            last.feedback["new_public_flips"] = flips
        if "oracle_refreshes" in last.feedback:
            last.feedback["oracle_refreshes"] = oracle_refreshes
        if public_flips:
            last.feedback["public_flips"] = public_flips
        if "trace_tail" in last.feedback:
            last.feedback["trace_tail"] = trace_tail[-ISSUE_TRACE_TAIL_CAP:]
        if self._trace_disclosed():
            last.feedback["timing_digest"] = digest.as_dict()
        return last

    def _trace_disclosed(self) -> bool:
        """Whether the issued-event trace/timing is disclosed at this feedback level.

        Mirrors :meth:`Disclosure.project_trace`: only ``full_trace`` echoes the
        trace, so the ``timing_digest`` (a summary of that trace) is gated the same
        way. The bare env (no resolver) discloses everything.
        """
        if self._resolver is None:
            return True
        return self._resolver.disclosure.feedback not in ("summarized_counts", "reward_only")

    def _digest_addr_key(self, command: dict[str, Any]) -> str:
        """A leak-safe ``per_addr_hits`` key: the address token the policy supplied.

        A handle's linear address is hidden, so the key is the handle *id* the
        policy already holds; a logical/bare-int address is itself what the policy
        supplied (== its linear address). Physical is only ever supplied under
        physical disclosure, where the linear address is publicly computable.
        """
        supplied = command.get("addr", command)
        if isinstance(supplied, dict):
            kind = supplied.get("kind")
            if kind == "handle":
                return f"handle:{supplied.get('id')}"
            if kind == "logical":
                return str(supplied.get("addr"))
        if isinstance(supplied, int) and not isinstance(supplied, bool):
            return str(supplied)
        return str(self._addr_value(command))

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
