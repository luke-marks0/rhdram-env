"""Plain records a rollout produces, plus aggregation with confidence intervals.

Torch-free on purpose: the rollout driver, the reference/controls, the artifact
writer, and the tests all use these without pulling in the model stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    """One assistant turn: what the policy said and what the environment returned.

    ``prompt_ids``/``gen_ids``/``gen_logprobs`` are populated only for a
    token-capturing (model-backed) policy; scripted policies leave them empty and
    the GRPO trainer simply never sees those rollouts.
    """

    tool: str
    args: dict[str, Any]
    text: str
    accepted: bool
    error: str | None
    feedback: dict[str, Any] = field(default_factory=dict)
    public_counters: dict[str, Any] = field(default_factory=dict)
    prompt_ids: list[int] = field(default_factory=list)
    gen_ids: list[int] = field(default_factory=list)
    gen_logprobs: list[float] = field(default_factory=list)

    def as_public(self) -> dict[str, Any]:
        """Disclosed-trajectory row for the JSONL artifact (no token ids)."""
        return {
            "tool": self.tool,
            "args": self.args,
            "accepted": self.accepted,
            "error": self.error,
            "feedback": self.feedback,
            "public_counters": self.public_counters,
        }


@dataclass
class EpisodeResult:
    task: str
    stage: str
    seed: int
    success: float          # trusted sparse reward, 0.0/1.0
    shaping: float          # bounded probe shaping (training only), already weighted
    turns: list[TurnRecord]
    decisive_probes: int
    unique_probes: int
    valid_calls: int
    total_calls: int
    budget_used: dict[str, int]
    error: str | None
    done_reason: str

    @property
    def reward(self) -> float:
        """Total training reward: sparse success plus bounded shaping."""
        return self.success + self.shaping

    @property
    def valid_call_rate(self) -> float:
        return self.valid_calls / self.total_calls if self.total_calls else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "stage": self.stage,
            "seed": self.seed,
            "success": self.success,
            "shaping": self.shaping,
            "reward": self.reward,
            "turns": len(self.turns),
            "decisive_probes": self.decisive_probes,
            "unique_probes": self.unique_probes,
            "valid_calls": self.valid_calls,
            "total_calls": self.total_calls,
            "valid_call_rate": self.valid_call_rate,
            "budget_used": self.budget_used,
            "error": self.error,
            "done_reason": self.done_reason,
        }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate + Wilson score interval for a success rate.

    Wilson (not normal-approximation) because success rates near 0 or 1 with the
    ~32-seed samples this PoC reports otherwise produce intervals that run past
    [0, 1] and understate uncertainty.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def aggregate(results: list[EpisodeResult]) -> dict[str, Any]:
    """Condition-level metrics for the writeup: success rate + CI and the rest."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    successes = sum(1 for r in results if r.success >= 1.0)
    point, lo, hi = wilson_interval(successes, n)
    budget_exhausted = sum(1 for r in results if r.error == "BUDGET_EXCEEDED")
    errored = sum(1 for r in results if r.error and r.error != "BUDGET_EXCEEDED")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "n": n,
        "success_rate": point,
        "success_ci95": [lo, hi],
        "success_count": successes,
        "mean_reward": mean([r.reward for r in results]),
        "mean_turns": mean([float(len(r.turns)) for r in results]),
        "mean_decisive_probes": mean([float(r.decisive_probes) for r in results]),
        "mean_unique_probes": mean([float(r.unique_probes) for r in results]),
        "valid_call_rate": mean([r.valid_call_rate for r in results]),
        "mean_acts_used": mean([float(r.budget_used.get("acts", 0)) for r in results]),
        "mean_cycles_used": mean([float(r.budget_used.get("cycles", 0)) for r in results]),
        "mean_tool_calls_used": mean([float(r.budget_used.get("tool_calls", 0)) for r in results]),
        "budget_exhausted_rate": budget_exhausted / n,
        "error_rate": errored / n,
    }
