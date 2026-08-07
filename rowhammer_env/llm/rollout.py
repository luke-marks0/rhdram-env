from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from rowhammer_env import Phase2Action
from rowhammer_env.observability.metrics import EpisodeResult, TrajectoryStep

from .policies import ToolCall, ToolPolicy

# ``RowHammerClient`` (the HTTP/websocket path used by ``run_episode``) is imported
# lazily inside the async entrypoints: it pulls in the OpenEnv websocket runtime,
# which is absent on the pure-Python host where the P26 reference-policy checks run
# via ``run_episode_local``. Keeping the import local lets this module — and the
# in-process driver — import without the P17 serving dependencies.


@dataclass(frozen=True)
class RolloutConfig:
    base_url: str
    seed: int
    task: dict[str, Any] | None = None
    episode_id: str | None = None
    max_steps: int = 4


async def run_episode(config: RolloutConfig, policy: ToolPolicy) -> EpisodeResult:
    from rowhammer_env.client import RowHammerClient

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


def run_episode_local(
    env: Any,
    policy: ToolPolicy,
    *,
    seed: int,
    task: dict[str, Any] | None = None,
    episode_id: str | None = None,
    budgets: dict[str, Any] | None = None,
    max_steps: int = 512,
) -> EpisodeResult:
    """Drive ``policy`` against an in-process env, mirroring :func:`run_episode`.

    Identical turn-by-turn loop to :func:`run_episode` — ``policy.next_tool`` once
    per real ``step()`` until ``done``/``max_steps`` — but calling a
    ``RowHammerTaskEnv`` directly instead of over HTTP via :class:`RowHammerClient`.
    This is the host-runnable analog used by the P26 reference-policy checks (the
    host has no websockets server), and drives the *same* ``ToolPolicy`` contract an
    LLM policy uses, so the trajectory is trace-equivalent to the HTTP path.
    """
    trajectory: list[TrajectoryStep] = []
    transcript: list[dict[str, Any]] = []
    reset_kwargs: dict[str, Any] = {"seed": seed, "episode_id": episode_id}
    if task is not None:
        reset_kwargs["task"] = task
    if budgets is not None:
        reset_kwargs["budgets"] = budgets
    observation = env.reset(**reset_kwargs)
    last_reward = float(observation.reward or 0.0)
    done = bool(observation.done)
    # The terminal (episode.finish) observation carries no task metadata; keep the
    # richest metadata seen so EpisodeResult's family/difficulty/budget stay set.
    metadata: dict[str, Any] = dict(observation.metadata or {})
    for _ in range(max_steps):
        if done:
            break
        call = policy.next_tool(observation, transcript)
        observation = env.step(_action(call))
        last_reward = float(observation.reward or 0.0)
        done = bool(observation.done)
        if observation.metadata:
            metadata = dict(observation.metadata)
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
    return EpisodeResult.from_rollout(
        seed=seed,
        task=task,
        final_observation=observation,
        reward=last_reward,
        done=done,
        trajectory=trajectory,
        metadata=metadata,
    )


def _action(call: ToolCall) -> Phase2Action:
    return Phase2Action(tool=call.name, args=dict(call.args))
