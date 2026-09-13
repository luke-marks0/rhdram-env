from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def observation_dict(observation: Any) -> dict[str, Any]:
    """Only the already-disclosed observation; never inspect environment internals."""
    if hasattr(observation, "model_dump"):
        data = observation.model_dump()
        data["metadata"] = dict(getattr(observation, "metadata", {}) or {})
        data.pop("info", None)  # duplicate wire mirror
        return data
    return {
        name: getattr(observation, name, default)
        for name, default in (("cycle", 0), ("reward", 0.0), ("done", False),
                              ("error", None), ("metadata", {}), ("feedback", {}),
                              ("public_counters", {}), ("last_action", {}))
    }


@dataclass(frozen=True)
class TrajectoryStep:
    action: dict[str, Any]
    reward: float
    done: bool
    error: dict[str, str] | None
    cycle: int
    feedback: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    assistant_text: str | None = None
    parse_valid: bool = True
    driver_generated: bool = False

    def as_public(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "error": self.error,
            "cycle": self.cycle,
        }


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    task_id: str
    family: str
    difficulty: str
    split: str | None
    reward: float
    done: bool
    success: bool
    steps: int
    final_cycle: int
    budget_remaining: dict[str, Any]
    trajectory: list[TrajectoryStep]
    final_observation: Any = field(repr=False, compare=False)
    initial_observation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rollout(
        cls,
        *,
        seed: int,
        task: dict[str, Any] | None,
        final_observation: Any,
        reward: float,
        done: bool,
        trajectory: list[TrajectoryStep],
        metadata: dict[str, Any],
        initial_observation: dict[str, Any] | None = None,
    ) -> "EpisodeResult":
        objective = (task or {}).get("objective") or {}
        return cls(
            seed=seed,
            task_id=str(metadata.get("task_id") or (task or {}).get("id") or "unknown"),
            family=str(metadata.get("task_family") or (task or {}).get("family") or "unknown"),
            difficulty=str((metadata.get("difficulty") or {}).get("band") or (task or {}).get("difficulty") or "unknown"),
            split=objective.get("split") or (task or {}).get("split"),
            reward=float(reward),
            done=bool(done),
            success=float(reward) == 1.0,
            steps=len(trajectory),
            final_cycle=int(getattr(final_observation, "cycle", 0)),
            budget_remaining=dict(metadata.get("budget_remaining") or {}),
            trajectory=list(trajectory),
            final_observation=final_observation,
            initial_observation=initial_observation or {},
        )

    def as_metrics_input(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "task_id": self.task_id,
            "family": self.family,
            "difficulty": self.difficulty,
            "split": self.split,
            "reward": self.reward,
            "done": self.done,
            "success": self.success,
            "steps": self.steps,
            "final_cycle": self.final_cycle,
            "budget_remaining": self.budget_remaining,
        }


@dataclass(frozen=True)
class EvalMetrics:
    episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    mean_final_cycle: float
    budget_efficiency: float
    by_family: dict[str, float]
    by_difficulty: dict[str, float]
    by_split: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "mean_steps": self.mean_steps,
            "mean_final_cycle": self.mean_final_cycle,
            "budget_efficiency": self.budget_efficiency,
            "by_family": self.by_family,
            "by_difficulty": self.by_difficulty,
            "by_split": self.by_split,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def summarize_episodes(results: list[EpisodeResult]) -> EvalMetrics:
    n = len(results)
    if n == 0:
        return EvalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}, {})
    successes = sum(1 for r in results if r.success)
    mean_reward = sum(r.reward for r in results) / n
    mean_steps = sum(r.steps for r in results) / n
    mean_final_cycle = sum(r.final_cycle for r in results) / n
    successful_steps = [r.steps for r in results if r.success and r.steps > 0]
    budget_efficiency = (successes / sum(successful_steps)) if successful_steps else 0.0
    return EvalMetrics(
        episodes=n,
        success_rate=successes / n,
        mean_reward=mean_reward,
        mean_steps=mean_steps,
        mean_final_cycle=mean_final_cycle,
        budget_efficiency=budget_efficiency,
        by_family=_group_rate(results, "family"),
        by_difficulty=_group_rate(results, "difficulty"),
        by_split=_group_rate(results, "split"),
    )


def _group_rate(results: list[EpisodeResult], attr: str) -> dict[str, float]:
    buckets: dict[str, list[EpisodeResult]] = {}
    for result in results:
        key = getattr(result, attr) or "none"
        buckets.setdefault(str(key), []).append(result)
    return {key: sum(1 for item in items if item.success) / len(items) for key, items in sorted(buckets.items())}
