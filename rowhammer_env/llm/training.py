from __future__ import annotations

from typing import Any

from rowhammer_env.observability.metrics import EpisodeResult

from .policies import TrainableHammerPolicy
from .rollout import RolloutConfig, run_episode


async def train_hammer_pairs(
    *,
    base_url: str,
    task: dict[str, Any] | None,
    seed: int,
    episodes: int = 2,
    initial_pairs: int = 1,
) -> dict[str, Any]:
    policy = TrainableHammerPolicy(initial_pairs=initial_pairs)
    updates: list[dict[str, Any]] = []
    results: list[EpisodeResult] = []
    for idx in range(episodes):
        policy._used = False
        result = await run_episode(
            RolloutConfig(base_url=base_url, seed=seed, task=task, episode_id=f"p19_train_{idx}", max_steps=2),
            policy,
        )
        results.append(result)
        updates.append(policy.update(reward=result.reward, observation=result.final_observation))
    return {
        "updates": updates,
        "rewards": [r.reward for r in results],
        "pair_counts": [u["pairs_after"] for u in updates],
        "improved": any(u["pairs_after"] > u["pairs_before"] for u in updates),
    }
