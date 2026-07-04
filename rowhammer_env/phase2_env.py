from __future__ import annotations

import base64
import binascii
import pathlib
from typing import Any

from pydantic import Field

from .openenv_source import load_openenv_server_types
from .tasks import AddressResolver
from .tools.addressing import AddressError
from .worker_protocol import WorkerClient, WorkerRequest


ROOT = pathlib.Path(__file__).resolve().parents[1]
Environment, Action, Observation, State = load_openenv_server_types()


class Phase2Action(Action):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Phase2Observation(Observation):
    cycle: int = 0
    data_b64: str | None = None
    last_action: dict[str, Any] = Field(default_factory=dict)
    public_counters: dict[str, Any] = Field(default_factory=dict)
    feedback: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, str] | None = None


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
            },
        )

    def step(self, action: Phase2Action, timeout_s: float | None = None, **_: Any) -> Phase2Observation:
        del timeout_s
        self._state.step_count += 1
        try:
            if action.tool == "dram.info":
                forms = sorted(self._resolver.disclosure.allowed_forms()) if self._resolver else ["logical"]
                return Phase2Observation(
                    reward=0.0,
                    done=False,
                    cycle=self._state.cycle,
                    metadata={"address_forms": forms, "commands": ["RD", "WR", "WAIT"]},
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
        last = Phase2Observation(reward=0.0, done=False, cycle=self._state.cycle)
        for command in commands:
            op = command.get("op")
            if op == "WAIT":
                req = WorkerRequest("ISSUE", self._next_id(), ("WAIT", str(command.get("cycles", 0))))
            elif op == "RD":
                req = WorkerRequest("ISSUE", self._next_id(), ("RD", str(self._addr_value(command))))
            elif op == "WR":
                raw = base64.b64decode(str(command.get("data_b64", "")), validate=True)
                req = WorkerRequest("ISSUE", self._next_id(), ("WR", str(self._addr_value(command)), raw.hex()))
            else:
                return self._error("ILLEGAL_COMMAND", str(op))
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
