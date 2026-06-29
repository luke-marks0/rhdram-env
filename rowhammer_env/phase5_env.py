from __future__ import annotations

from typing import Any

from .phase2_env import Phase2Action, Phase2Observation
from .phase4_env import RowHammerDisturbanceEnv
from .script_sandbox import RestrictedScriptBroker, ScriptViolation


class RowHammerTaskEnv(RowHammerDisturbanceEnv):
    def __init__(
        self,
        *args: Any,
        budgets: dict[str, int] | None = None,
        task: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.initial_budgets = budgets or {"tool_calls": 20000, "cycles": 5_000_000}
        self.budget_remaining = dict(self.initial_budgets)
        self.task_config = task or {"family": "known_target_anybit"}
        self.task_family = self.task_config["family"]
        self.target_row = 10
        self.success = False

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Phase2Observation:
        self.budget_remaining = dict(self.initial_budgets)
        self.success = False
        self.task_family = self.task_config.get("family", "known_target_anybit")
        obs = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        if obs.error:
            return obs
        self.target_row = self.disturbance.known_target_row  # type: ignore[union-attr]
        obs.metadata.update(self._task_metadata(seed))
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
        if self.task_family == "any_flip":
            return bool(self.disturbance.flips)
        target_addr = self.target_row * self.disturbance.row_bytes
        if self.task_family == "target_cell":
            return self.disturbance.flips.get(target_addr) == int(self.task_config.get("bit", 0))
        if self.task_family == "pattern_target":
            desired = int(self.task_config.get("value", 1))
            return self.disturbance.flips.get(target_addr) == 0 and (desired & 1) == 1
        return any(addr // self.disturbance.row_bytes == self.target_row for addr in self.disturbance.flips)

    def _task_metadata(self, seed: int | None) -> dict[str, Any]:
        assert self.disturbance is not None
        target_addr = self.disturbance.target_addr
        public_disturbance = {
            "family": self.disturbance.family,
            "stratum": self.disturbance.stratum,
        }
        if self.task_family not in {"hidden_target", "unknown_adjacency"}:
            public_disturbance.update(
                {
                    "known_target_row": self.disturbance.known_target_row,
                    "known_threshold": self.disturbance.known_threshold,
                }
            )

        meta = {
            "task_id": f"ddr4_{self.task_family}_v1",
            "task_family": self.task_family,
            "difficulty": {"band": "smoke", "seed": seed},
            "budget_remaining": dict(self.budget_remaining),
            "allowed_tools": [
                "dram.info",
                "dram.read",
                "dram.write",
                "dram.issue",
                "script.run",
                "episode.finish",
            ],
            "disclosure": {"mapping": "logical_only", "feedback": "summarized_counts"},
            "disturbance": public_disturbance,
        }

        if self.task_family == "any_flip":
            meta["objective"] = {"type": "any_flip"}
        elif self.task_family == "target_cell":
            meta["objective"] = {"type": "target_cell_flip", "bit": int(self.task_config.get("bit", 0))}
            meta["target"] = {"kind": "logical", "addr": target_addr}
        elif self.task_family == "pattern_target":
            meta["objective"] = {
                "type": "pattern_target",
                "mask": int(self.task_config.get("mask", 1)),
                "value": int(self.task_config.get("value", 1)),
            }
            meta["target"] = {"kind": "logical", "addr": target_addr}
        elif self.task_family == "hidden_target":
            meta["objective"] = {"type": "target_row_flip", "target": {"kind": "handle", "id": "target:0"}}
            meta["disclosure"].update({"adjacency": "hidden", "victim": "row_handle"})
        elif self.task_family == "unknown_adjacency":
            meta["objective"] = {"type": "target_row_flip", "target": {"kind": "handle", "id": "target:0"}}
            meta["disclosure"].update({"adjacency": "candidate_set", "victim": "row_handle"})
            rb = self.disturbance.row_bytes
            meta["candidates"] = [{"kind": "logical", "addr": target_addr + off} for off in (-rb, rb, rb * 2)]
        else:
            meta["objective"] = {"type": "target_row_flip", "target_row": self.target_row}
            meta["target"] = {"kind": "logical", "addr": target_addr}

        return meta
