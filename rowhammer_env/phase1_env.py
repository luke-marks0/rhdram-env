from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

from pydantic import Field

from .openenv_source import load_openenv_server_types


ROOT = pathlib.Path(__file__).resolve().parents[1]
Environment, Action, Observation, State = load_openenv_server_types()


class Phase1Action(Action):
    tool: str = Field(default="dram.bootstrap")


class Phase1Observation(Observation):
    cycle: int = 0
    ramulator: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, str] | None = None


class Phase1State(State):
    ramulator_bootstrap_complete: bool = False


class RowHammerBootstrapEnv(Environment[Phase1Action, Phase1Observation, Phase1State]):
    def __init__(
        self,
        worker_path: pathlib.Path | None = None,
        config_path: pathlib.Path | None = None,
    ) -> None:
        super().__init__()
        self.worker_path = worker_path or ROOT / "build/phase1/ramulator_external_smoke"
        self.config_path = config_path or ROOT / "build/phase1/p1_external_ddr4.yaml"
        self._state = Phase1State(episode_id=None, step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **_: Any,
    ) -> Phase1Observation:
        self._state = Phase1State(
            episode_id=episode_id or "p1_bootstrap",
            step_count=0,
            ramulator_bootstrap_complete=False,
        )
        return Phase1Observation(
            reward=0.0,
            done=False,
            metadata={"seed": seed, "allowed_tools": ["dram.bootstrap"]},
        )

    def step(
        self,
        action: Phase1Action,
        timeout_s: float | None = None,
        **_: Any,
    ) -> Phase1Observation:
        self._state.step_count += 1
        if action.tool != "dram.bootstrap":
            return self._error("UNSUPPORTED_TOOL", action.tool)
        if not self.worker_path.is_file() or not self.config_path.is_file():
            return self._error("UNAVAILABLE_CAPABILITY", "Ramulator bootstrap is not built")

        result = subprocess.run(
            [str(self.worker_path), str(self.config_path)],
            text=True,
            capture_output=True,
            timeout=timeout_s or 10.0,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return self._error("INTERNAL_SIMULATOR_ERROR", result.stderr.strip())

        if result.returncode != 0 or not payload.get("ok"):
            err = payload.get("error") or {}
            return self._error(
                err.get("code", "INTERNAL_SIMULATOR_ERROR"),
                err.get("message", "Ramulator bootstrap failed"),
            )

        self._state.ramulator_bootstrap_complete = True
        return Phase1Observation(
            cycle=int(payload["ticks"]),
            ramulator=payload,
            reward=0.0,
            done=False,
        )

    @property
    def state(self) -> Phase1State:
        return self._state

    def _error(self, code: str, message: str) -> Phase1Observation:
        return Phase1Observation(
            reward=0.0,
            done=True,
            error={"code": code, "message": message},
        )
