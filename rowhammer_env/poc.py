"""The deliberately narrow, real-simulator PoC runtime and experiment contract."""
from __future__ import annotations

import copy
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


def seed_list(spec: dict[str, Any]) -> list[int]:
    """Resolve one explicit seed list or an inclusive range, rejecting ambiguity."""
    if ("seeds" in spec) == ("seed_range" in spec):
        raise ValueError("give exactly one of seeds or seed_range")
    if "seed_range" in spec:
        bounds = spec["seed_range"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("seed_range must be [start, stop]")
        if any(type(n) is not int for n in bounds) or bounds[1] < bounds[0]:
            raise ValueError("seed_range needs ordered integer bounds")
        seeds = list(range(bounds[0], bounds[1] + 1))
    else:
        seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(n) is not int for n in seeds):
        raise ValueError("seeds must be a nonempty integer list")
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seeds would overweight episodes")
    return list(seeds)


def validate_config(cfg: dict[str, Any]) -> None:
    from .llm.curriculum import load_curriculum
    from .llm.shaping import shaping_weights

    stages = load_curriculum(cfg)
    paths = [path for stage in stages for path in stage.tasks]
    if paths != list(TASK_PATHS):
        raise ValueError("PoC curriculum must contain exactly the five scoped tasks in order")
    for path in paths:
        load_task(path)
    train = set().union(*(set(stage.seeds) for stage in stages))
    validation = set(seed_list(cfg["eval"]))
    test = set(seed_list(cfg["benchmark"]))
    if train & validation or train & test or validation & test:
        raise ValueError("training, validation, and benchmark seeds must be disjoint")
    if len(test) < 32:
        raise ValueError("PoC benchmark needs at least 32 held-out seeds")
    if "probe_shaping_weight" not in cfg.get("reward", {}):
        raise ValueError("record probe_shaping_weight explicitly, including zero")
    shaping_weights(cfg["reward"])
    if not cfg.get("rollout", {}).get("multi_turn"):
        raise ValueError("PoC requires multi-turn rollouts")
    if cfg.get("model", {}).get("enable_thinking"):
        raise ValueError("the supported PoC recipe uses action-only, non-thinking responses")
    if int(cfg["rollout"].get("max_turns", 0)) < 33:
        raise ValueError("medium reference discovery needs at least 33 turns")


def resolved_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["resolved_tasks"] = {p: load_task(p) for p in TASK_PATHS}
    out["resolved_validation_seeds"] = seed_list(cfg["eval"])
    out["resolved_benchmark_seeds"] = seed_list(cfg["benchmark"])
    return out
