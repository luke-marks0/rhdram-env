"""The deliberately narrow, real-simulator PoC environment."""
from __future__ import annotations

import pathlib
from typing import Any

import yaml

from .phase2_env import Phase2Action, Phase2Observation
from .phase5_env import RowHammerTaskEnv
from .tasks.compiler import TaskConfigError, TaskSpec

ROOT = pathlib.Path(__file__).resolve().parents[1]
POC_TOOLS = ("dram.info", "dram.issue", "episode.finish")
TASK_PATHS = (
    "configs/tasks/known_target_anybit.yaml",
    "configs/tasks/bounded_sweep_easy.yaml",
    "configs/tasks/bounded_sweep_medium.yaml",
    "configs/tasks/hidden_adjacency_easy.yaml",
    "configs/tasks/hidden_adjacency_medium.yaml",
)


def load_task(path: str) -> dict[str, Any]:
    task = yaml.safe_load((ROOT / path).read_text())
    validate_task(task)
    return task


def validate_task(task: dict[str, Any] | None) -> None:
    spec = TaskSpec.from_config(task)
    if spec.family not in {"known_target_anybit", "bounded_sweep", "hidden_adjacency"}:
        raise TaskConfigError("task family is outside the PoC")
    if spec.family != "known_target_anybit" and spec.difficulty not in {"easy", "medium"}:
        raise TaskConfigError("PoC discovery supports easy and medium only")
    if (task or {}).get("standard", "DDR4") != "DDR4":
        raise TaskConfigError("PoC requires DDR4")
    if spec.profile_id not in (None, "ddr4_vts25_v1"):
        raise TaskConfigError("PoC requires ddr4_vts25_v1")
    if spec.mitigation.get("name") != "none":
        raise TaskConfigError("PoC requires mitigation: none")


class PoCEnv(RowHammerTaskEnv):
    """Same trusted environment, with a pre-dispatch three-tool admission gate."""

    def __init__(self, *, task: dict[str, Any] | None = None, **kwargs: Any) -> None:
        validate_task(task)
        if kwargs.get("temperature", 50) != 50:
            raise ValueError("PoC requires 50 C")
        if kwargs.get("profile_id", "ddr4_vts25_v1") != "ddr4_vts25_v1":
            raise ValueError("PoC requires ddr4_vts25_v1")
        if kwargs.get("mitigation", {"name": "none"}).get("name") != "none":
            raise ValueError("PoC requires mitigation: none")
        super().__init__(task=task, **kwargs)

    def reset(self, *args: Any, task: dict[str, Any] | None = None, **kwargs: Any) -> Phase2Observation:
        try:
            validate_task(task)
        except TaskConfigError as exc:
            return self._refuse("BAD_SCHEMA", str(exc))
        obs = super().reset(*args, task=task, **kwargs)
        if not obs.error and (self.geometry.standard != "DDR4" or self.geometry.level_sizes["channel"] != 1):
            return self._refuse("UNAVAILABLE_CAPABILITY", "PoC requires single-channel DDR4")
        return obs

    def _task_metadata(self, seed: int | None) -> dict[str, Any]:
        return {**super()._task_metadata(seed), "allowed_tools": list(POC_TOOLS), "policy_surface": "poc"}

    def step(self, action: Phase2Action, **kwargs: Any) -> Phase2Observation:
        obs = super().step(action, **kwargs)
        obs.metadata.update(allowed_tools=list(POC_TOOLS), policy_surface="poc")
        return obs

    def _dispatch(self, action: Phase2Action, **kwargs: Any) -> Phase2Observation:
        if action.tool and action.tool not in POC_TOOLS:
            obs = self._error("UNSUPPORTED_TOOL", "tool is outside the PoC policy surface")
            obs.public_counters = dict(self._public_counters)
            self._charge(obs, self._state.cycle)
            obs.metadata["budget_remaining"] = dict(self.budget_remaining)
            return obs
        return super()._dispatch(action, **kwargs)
