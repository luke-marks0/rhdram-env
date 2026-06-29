from __future__ import annotations

from typing import Any

from .phase2_env import Phase2Action, Phase2Observation
from .phase4_env import RowHammerDisturbanceEnv
from .script_sandbox import RestrictedScriptBroker, ScriptViolation


class RowHammerTaskEnv(RowHammerDisturbanceEnv):
    def __init__(self, *args: Any, budgets: dict[str, int] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.initial_budgets = budgets or {"tool_calls": 20000, "cycles": 5_000_000}
        self.budget_remaining = dict(self.initial_budgets)
        self.target_row = 10
        self.success = False

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Phase2Observation:
        self.budget_remaining = dict(self.initial_budgets)
        self.success = False
        obs = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        if obs.error:
            return obs
        self.target_row = self.disturbance.known_target_row  # type: ignore[union-attr]
        obs.metadata.update(
            {
                "task_id": "ddr4_known_target_anybit_v1",
                "objective": {"type": "target_row_flip", "target_row": self.target_row},
                "target": {"kind": "logical", "addr": self.disturbance.target_addr},  # type: ignore[union-attr]
                "budget_remaining": dict(self.budget_remaining),
                "allowed_tools": [
                    "dram.info",
                    "dram.read",
                    "dram.write",
                    "dram.issue",
                    "script.run",
                    "episode.finish",
                ],
            }
        )
        return obs

    def step(self, action: Phase2Action, timeout_s: float | None = None, **kwargs: Any) -> Phase2Observation:
        if action.tool == "script.run":
            return self._script(action)
        if action.tool == "episode.finish":
            obs = Phase2Observation(reward=1.0 if self._trusted_success() else 0.0, done=True, cycle=self._state.cycle)
            self.close()
            return obs

        before = self._state.cycle
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        self._charge(obs, before)
        self.success = self._trusted_success()
        if self.success:
            obs.reward = 1.0
            obs.done = True
        obs.metadata["budget_remaining"] = dict(self.budget_remaining)
        return obs

    def _script(self, action: Phase2Action) -> Phase2Observation:
        code = str(action.args.get("code", ""))
        try:
            result = RestrictedScriptBroker(self, max_calls=min(10_000, self.budget_remaining["tool_calls"])).run(code)
        except ScriptViolation as exc:
            return self._error(exc.code, str(exc))
        obs = Phase2Observation(reward=1.0 if self.success else 0.0, done=self.success, cycle=self._state.cycle)
        obs.feedback["script"] = result
        obs.metadata["budget_remaining"] = dict(self.budget_remaining)
        return obs

    def _charge(self, obs: Phase2Observation, before_cycle: int) -> None:
        self.budget_remaining["tool_calls"] -= 1
        self.budget_remaining["cycles"] -= max(0, obs.cycle - before_cycle)
        if self.budget_remaining["tool_calls"] < 0 or self.budget_remaining["cycles"] < 0:
            obs.error = {"code": "BUDGET_EXCEEDED", "message": "episode budget exhausted"}
            obs.done = True

    def _trusted_success(self) -> bool:
        if self.disturbance is None:
            return False
        return any(addr // self.disturbance.row_bytes == self.target_row for addr in self.disturbance.flips)
