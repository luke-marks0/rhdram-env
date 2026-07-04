from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from rowhammer_env import Phase2Action
from rowhammer_env.client import RowHammerClient
from rowhammer_env.observability.metrics import EpisodeResult, TrajectoryStep

from .policies import ToolCall, ToolPolicy


@dataclass(frozen=True)
class RolloutConfig:
    base_url: str
    seed: int
    task: dict[str, Any] | None = None
    episode_id: str | None = None
    max_steps: int = 4


async def run_episode(config: RolloutConfig, policy: ToolPolicy) -> EpisodeResult:
    trajectory: list[TrajectoryStep] = []
    transcript: list[dict[str, Any]] = []
    async with RowHammerClient(base_url=config.base_url, message_timeout_s=120.0) as client:
        reset = await client.reset(seed=config.seed, episode_id=config.episode_id, task=config.task)
        observation = reset.observation
        last_reward = float(reset.reward or 0.0)
        done = bool(reset.done)
        for _ in range(config.max_steps):
            if done:
                break
            call = policy.next_tool(observation, transcript)
            result = await client.step(_action(call))
            observation = result.observation
            last_reward = float(result.reward or 0.0)
            done = bool(result.done)
            step = TrajectoryStep(
                action={"tool": call.name, "args": call.args},
                reward=last_reward,
                done=done,
                error=observation.error,
                cycle=observation.cycle,
                feedback=observation.feedback,
            )
            trajectory.append(step)
            transcript.append(step.as_public())
            if done:
                break
        metadata = observation.metadata
        return EpisodeResult.from_rollout(
            seed=config.seed,
            task=config.task,
            final_observation=observation,
            reward=last_reward,
            done=done,
            trajectory=trajectory,
            metadata=metadata,
        )


async def run_curriculum(
    *,
    base_url: str,
    policy_factory: Callable[[], ToolPolicy],
    tasks: Iterable[dict[str, Any] | None],
    seeds: Iterable[int],
    max_steps: int = 4,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for task in tasks:
        for seed in seeds:
            config = RolloutConfig(base_url=base_url, seed=int(seed), task=task, episode_id=f"p19_{seed}", max_steps=max_steps)
            results.append(await run_episode(config, policy_factory()))
    return results


def run_curriculum_sync(**kwargs: Any) -> list[EpisodeResult]:
    return asyncio.run(run_curriculum(**kwargs))


def _action(call: ToolCall) -> Phase2Action:
    return Phase2Action(tool=call.name, args=dict(call.args))
