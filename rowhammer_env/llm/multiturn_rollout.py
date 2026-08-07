"""Multi-turn training rollout + completion-mask assembly (P27).

The eval path (``rollout.run_episode`` / ``run_episode_local``) is already
turn-by-turn interactive; the single-shot GRPO path instead parses one completion
into an action list and replays it. This module is the torch/trl-free multi-turn
training loop plus the completion-mask builder; the trl/torch binding lives in
``scripts/train_grpo.py`` so this stays importable and host-testable.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from rowhammer_env import Phase2Action
from rowhammer_env.observability.metrics import EpisodeResult, TrajectoryStep

from .grpo_env import build_messages, parse_actions
from .policies import ToolCall, ToolPolicy

# Masking keys off role == "assistant", so any non-assistant role works here; "tool" is
# the semantically correct one for templates that expose it (Qwen3 does). Map to "user"
# at render time for a template without one.
TOOL_ROLE = "tool"

# full_trace's trace_tail repeats 50+ near-identical rows in every turn, bloating each
# subsequent prompt with no signal the digest (acts_delta/per_addr_hits) doesn't already
# carry. Cap how many rows reach the model; 0 drops it entirely.
_MAX_TRACE_TAIL = 4


# --------------------------------------------------------------------------- #
# Generator interface: turn context -> one assistant completion (text)
# --------------------------------------------------------------------------- #
class CompletionGenerator(Protocol):
    """Produce one assistant completion for the current turn.

    A model-backed generator uses ``messages``; a :class:`ToolPolicyGenerator` uses
    ``observation``/``transcript`` instead — both are passed so either kind of
    decision-maker drops in without changing the loop.
    """

    def __call__(
        self, messages: list[dict[str, str]], observation: Any, transcript: list[dict[str, Any]]
    ) -> str:
        ...


def render_tool_call(call: ToolCall) -> str:
    """Render a tool call as the fenced-JSON completion the system prompt expects.

    Round-trips through :func:`grpo_env.parse_actions`, so driving a
    :class:`ToolPolicyGenerator` this way is trace-equivalent to ``run_episode_local``.
    """
    body = json.dumps({"actions": [{"tool": call.name, "args": call.args}]})
    return "```json\n" + body + "\n```"


def _trim_trace_tail(feedback: Any, limit: int = _MAX_TRACE_TAIL) -> Any:
    """Cap any ``trace_tail`` list (top-level and inside ``timing_digest``) to the last
    ``limit`` rows, recording the true length as ``trace_tail_len``. Returns a copy —
    never mutates the observation's own feedback.
    """
    if not isinstance(feedback, dict):
        return feedback

    def _cap(container: dict) -> dict:
        tail = container.get("trace_tail")
        if not isinstance(tail, list) or len(tail) <= limit:
            return container
        out = dict(container)
        out["trace_tail_len"] = len(tail)
        out["trace_tail"] = tail[-limit:] if limit > 0 else []
        return out

    trimmed = _cap(dict(feedback))
    digest = trimmed.get("timing_digest")
    if isinstance(digest, dict):
        trimmed["timing_digest"] = _cap(digest)
    return trimmed


def render_tool_result(observation: Any) -> str:
    """Render the disclosed step observation as the tool-turn text fed back.

    Already leakage-guard-stripped by ``step()``; ``trace_tail`` is capped
    (:func:`_trim_trace_tail`) so it doesn't bloat later turns' prompts.
    """
    payload = {
        "cycle": int(getattr(observation, "cycle", 0) or 0),
        "reward": float(getattr(observation, "reward", 0.0) or 0.0),
        "done": bool(getattr(observation, "done", False)),
        "error": getattr(observation, "error", None),
        "last_action": getattr(observation, "last_action", {}) or {},
        "public_counters": getattr(observation, "public_counters", {}) or {},
        "feedback": _trim_trace_tail(getattr(observation, "feedback", {}) or {}),
    }
    return json.dumps(payload, sort_keys=True)


class ToolPolicyGenerator:
    """Adapt a :class:`ToolPolicy` into a :class:`CompletionGenerator`.

    Lets a deterministic reference policy drive the training loop and produce a
    trajectory identical to the eval loop's (the P27 trace-equivalence check).
    """

    def __init__(self, policy: ToolPolicy) -> None:
        self._policy = policy

    def __call__(
        self, messages: list[dict[str, str]], observation: Any, transcript: list[dict[str, Any]]
    ) -> str:
        del messages
        return render_tool_call(self._policy.next_tool(observation, transcript))


# --------------------------------------------------------------------------- #
# Rollout result
# --------------------------------------------------------------------------- #
@dataclass
class MultiTurnRollout:
    """One collected multi-turn trajectory ready for GRPO.

    ``messages`` is the whole chat transcript; ``prompt_messages`` is the fixed
    prefix (system + task). ``reward`` is the trusted whole-episode reward.
    """

    prompt_messages: list[dict[str, str]]
    messages: list[dict[str, str]]
    trajectory: list[TrajectoryStep]
    result: EpisodeResult
    reward: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def completion_messages(self) -> list[dict[str, str]]:
        return self.messages[len(self.prompt_messages) :]


def _action(call: ToolCall) -> Phase2Action:
    return Phase2Action(tool=call.name, args=dict(call.args))


def _decide(
    generator: CompletionGenerator,
    messages: list[dict[str, str]],
    observation: Any,
    transcript: list[dict[str, Any]],
) -> tuple[str, ToolCall]:
    """Generate one completion and parse it into a single tool call.

    Uses the same parse path the single-shot trainer uses; an unparseable
    completion degrades to ``episode.finish`` (reward 0).
    """
    text = generator(messages, observation, transcript)
    actions = parse_actions(text)
    call = actions[0] if actions else ToolCall("episode.finish", {})
    return text, call


def _append_step(
    call: ToolCall,
    observation: Any,
    reward: float,
    done: bool,
    trajectory: list[TrajectoryStep],
    transcript: list[dict[str, Any]],
) -> None:
    step = TrajectoryStep(
        action={"tool": call.name, "args": call.args},
        reward=reward,
        done=done,
        error=observation.error,
        cycle=observation.cycle,
        feedback=observation.feedback,
    )
    trajectory.append(step)
    transcript.append(step.as_public())


def run_training_episode_local(
    env: Any,
    generator: CompletionGenerator,
    *,
    seed: int,
    task: dict[str, Any] | None = None,
    episode_id: str | None = None,
    budgets: dict[str, Any] | None = None,
    max_turns: int = 64,
) -> MultiTurnRollout:
    """Collect a multi-turn trajectory against an in-process env.

    Mirrors :func:`rollout.run_episode_local` turn-for-turn, but also records the
    rendered chat transcript alongside the trajectory.
    """
    reset_kwargs: dict[str, Any] = {"seed": seed, "episode_id": episode_id}
    if task is not None:
        reset_kwargs["task"] = task
    if budgets is not None:
        reset_kwargs["budgets"] = budgets
    observation = env.reset(**reset_kwargs)
    return _drive(observation, env.step, generator, seed=seed, task=task, max_turns=max_turns)


async def run_training_episode(
    config: "RolloutConfig",
    generator: CompletionGenerator,
    *,
    max_turns: int = 64,
) -> MultiTurnRollout:
    """HTTP/WS twin of :func:`run_training_episode_local`, driving the real OpenEnv
    server via :class:`RowHammerClient`. Imports the client lazily (no websocket
    runtime on the host).
    """
    from rowhammer_env.client import RowHammerClient

    async with RowHammerClient(base_url=config.base_url, message_timeout_s=120.0) as client:
        reset = await client.reset(seed=config.seed, episode_id=config.episode_id, task=config.task)
        observation = reset.observation
        metadata = dict(getattr(observation, "metadata", {}) or {})
        prompt_messages = build_messages(metadata)
        messages = list(prompt_messages)
        transcript: list[dict[str, Any]] = []
        trajectory: list[TrajectoryStep] = []
        last_reward = float(reset.reward or 0.0)
        done = bool(reset.done)
        richest = dict(metadata)
        for _ in range(max_turns):
            if done:
                break
            text, call = _decide(generator, messages, observation, transcript)
            messages.append({"role": "assistant", "content": text})
            result = await client.step(_action(call))
            observation = result.observation
            last_reward = float(result.reward or 0.0)
            done = bool(result.done)
            if getattr(observation, "metadata", None):
                richest = dict(observation.metadata)
            _append_step(call, observation, last_reward, done, trajectory, transcript)
            if done:
                break
            messages.append({"role": TOOL_ROLE, "content": render_tool_result(observation)})
        episode = EpisodeResult.from_rollout(
            seed=config.seed,
            task=config.task,
            final_observation=observation,
            reward=last_reward,
            done=done,
            trajectory=trajectory,
            metadata=richest,
        )
        return MultiTurnRollout(
            prompt_messages=prompt_messages,
            messages=messages,
            trajectory=trajectory,
            result=episode,
            reward=last_reward,
            metadata=richest,
        )


def _drive(
    observation: Any,
    step_fn: Any,
    generator: CompletionGenerator,
    *,
    seed: int,
    task: dict[str, Any] | None,
    max_turns: int,
) -> MultiTurnRollout:
    """Shared synchronous loop body for the in-process driver."""
    metadata = dict(getattr(observation, "metadata", {}) or {})
    prompt_messages = build_messages(metadata)
    messages = list(prompt_messages)
    transcript: list[dict[str, Any]] = []
    trajectory: list[TrajectoryStep] = []
    last_reward = float(getattr(observation, "reward", 0.0) or 0.0)
    done = bool(getattr(observation, "done", False))
    richest = dict(metadata)
    for _ in range(max_turns):
        if done:
            break
        text, call = _decide(generator, messages, observation, transcript)
        messages.append({"role": "assistant", "content": text})
        observation = step_fn(_action(call))
        last_reward = float(getattr(observation, "reward", 0.0) or 0.0)
        done = bool(getattr(observation, "done", False))
        if getattr(observation, "metadata", None):
            richest = dict(observation.metadata)
        _append_step(call, observation, last_reward, done, trajectory, transcript)
        if done:
            break
        messages.append({"role": TOOL_ROLE, "content": render_tool_result(observation)})
    episode = EpisodeResult.from_rollout(
        seed=seed,
        task=task,
        final_observation=observation,
        reward=last_reward,
        done=done,
        trajectory=trajectory,
        metadata=richest,
    )
    return MultiTurnRollout(
        prompt_messages=prompt_messages,
        messages=messages,
        trajectory=trajectory,
        result=episode,
        reward=last_reward,
        metadata=richest,
    )


# --------------------------------------------------------------------------- #
# Batched multi-turn rollout: one batched generation per tick across many
# concurrent episodes (the throughput path for vLLM/HF batch generation, vs. the
# sequential single-episode loops above).
# --------------------------------------------------------------------------- #
@dataclass
class TurnRequest:
    """One episode's pending turn — the context a batched generator conditions on."""

    episode_id: str
    messages: list[dict[str, str]]
    observation: Any
    transcript: list[dict[str, Any]]


class BatchGenerator(Protocol):
    """Generate one completion per pending turn in a SINGLE batched call, aligned to
    ``requests`` (what makes vLLM/HF batching pay off vs. one episode at a time).
    """

    def __call__(self, requests: list[TurnRequest]) -> list[str]:
        ...


@dataclass
class _EpisodeState:
    """Mutable per-episode state carried across ticks of the batched driver."""

    seed: int
    task: dict[str, Any] | None
    episode_id: str
    observation: Any
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    last_reward: float = 0.0
    done: bool = False
    richest: dict[str, Any] = field(default_factory=dict)

    def start(self, *, reward: float | None = None, done: bool | None = None) -> None:
        metadata = dict(getattr(self.observation, "metadata", {}) or {})
        self.prompt_messages = build_messages(metadata)
        self.messages = list(self.prompt_messages)
        self.richest = dict(metadata)
        self.last_reward = float(getattr(self.observation, "reward", 0.0) or 0.0) if reward is None else float(reward)
        self.done = bool(getattr(self.observation, "done", False)) if done is None else bool(done)

    def request(self) -> TurnRequest:
        return TurnRequest(self.episode_id, self.messages, self.observation, self.transcript)

    def begin_turn(self, text: str) -> tuple[ToolCall, Phase2Action]:
        """Parse a completion, append it as the assistant turn, and return the tool call."""
        actions = parse_actions(text)
        call = actions[0] if actions else ToolCall("episode.finish", {})
        self.messages.append({"role": "assistant", "content": text})
        return call, _action(call)

    def end_turn(self, call: ToolCall, observation: Any, reward: Any, done: Any) -> None:
        """Fold one stepped observation back in — mirrors the single-episode loop body."""
        self.observation = observation
        self.last_reward = float(reward or 0.0)
        self.done = bool(done)
        if getattr(observation, "metadata", None):
            self.richest = dict(observation.metadata)
        _append_step(call, observation, self.last_reward, self.done, self.trajectory, self.transcript)
        if not self.done:
            self.messages.append({"role": TOOL_ROLE, "content": render_tool_result(observation)})

    def to_rollout(self) -> MultiTurnRollout:
        episode = EpisodeResult.from_rollout(
            seed=self.seed,
            task=self.task,
            final_observation=self.observation,
            reward=self.last_reward,
            done=self.done,
            trajectory=self.trajectory,
            metadata=self.richest,
        )
        return MultiTurnRollout(
            prompt_messages=self.prompt_messages,
            messages=self.messages,
            trajectory=self.trajectory,
            result=episode,
            reward=self.last_reward,
            metadata=self.richest,
        )


def run_batched_training_episodes_local(
    episodes: list[tuple[Any, int, dict[str, Any] | None, str | None]],
    batch_generator: BatchGenerator,
    *,
    max_turns: int = 64,
) -> list[MultiTurnRollout]:
    """Run several in-process episodes concurrently, one batched generation per tick.

    ``episodes`` is a list of ``(env, seed, task, episode_id)``. Host analog of
    :func:`run_batched_training_episodes`; with a per-episode
    :class:`BatchToolPolicyGenerator` this is trace-equivalent to running each
    episode through :func:`run_training_episode_local`.
    """
    states: list[tuple[_EpisodeState, Any]] = []
    for env, seed, task, episode_id in episodes:
        reset_kwargs: dict[str, Any] = {"seed": seed, "episode_id": episode_id}
        if task is not None:
            reset_kwargs["task"] = task
        obs = env.reset(**reset_kwargs)
        st = _EpisodeState(seed=seed, task=task, episode_id=episode_id or f"ep_{seed}", observation=obs)
        st.start()
        states.append((st, env))
    for _ in range(max_turns):
        active = [(st, env) for st, env in states if not st.done]
        if not active:
            break
        texts = batch_generator([st.request() for st, _ in active])
        for (st, env), text in zip(active, texts):
            call, action = st.begin_turn(text)
            obs = env.step(action)
            st.end_turn(call, obs, getattr(obs, "reward", 0.0), getattr(obs, "done", False))
    return [st.to_rollout() for st, _ in states]


async def run_batched_training_episodes(
    base_url: str,
    episodes: list[tuple[int, dict[str, Any] | None, str | None]],
    batch_generator: BatchGenerator,
    *,
    max_turns: int = 64,
    message_timeout_s: float = 120.0,
) -> list[MultiTurnRollout]:
    """Batched multi-turn rollout over the HTTP/WS transport (instance throughput path).

    ``episodes`` is ``(seed, task, episode_id)`` — one live env session each
    (``episode_id`` must be unique per concurrent rollout, even sharing a seed). Each
    tick renders every active episode's prompt, makes ONE batched ``batch_generator``
    call, then steps every env concurrently. Caller keeps the wave size within the
    server's ``max_concurrent_envs``.
    """
    from contextlib import AsyncExitStack

    from rowhammer_env.client import RowHammerClient

    async with AsyncExitStack() as stack:
        states: list[tuple[_EpisodeState, Any]] = []
        for seed, task, episode_id in episodes:
            client = await stack.enter_async_context(
                RowHammerClient(base_url=base_url, message_timeout_s=message_timeout_s)
            )
            reset = await client.reset(seed=seed, episode_id=episode_id, task=task)
            st = _EpisodeState(seed=seed, task=task, episode_id=episode_id or f"ep_{seed}", observation=reset.observation)
            st.start(reward=reset.reward, done=reset.done)
            states.append((st, client))
        for _ in range(max_turns):
            active = [(st, client) for st, client in states if not st.done]
            if not active:
                break
            texts = batch_generator([st.request() for st, _ in active])

            async def _advance(st: _EpisodeState, client: Any, text: str) -> None:
                call, action = st.begin_turn(text)
                result = await client.step(action)
                st.end_turn(call, result.observation, result.reward, result.done)

            await asyncio.gather(*[_advance(st, client, text) for (st, client), text in zip(active, texts)])
        return [st.to_rollout() for st, _ in states]


class BatchToolPolicyGenerator:
    """A :class:`BatchGenerator` backed by one :class:`ToolPolicy` per episode (host
    tests) — the batched analog of :class:`ToolPolicyGenerator`.
    """

    def __init__(self, policy_factory: Any) -> None:
        self._factory = policy_factory
        self._policies: dict[str, ToolPolicy] = {}

    def __call__(self, requests: list[TurnRequest]) -> list[str]:
        out: list[str] = []
        for req in requests:
            policy = self._policies.get(req.episode_id)
            if policy is None:
                policy = self._factory()
                self._policies[req.episode_id] = policy
            out.append(render_tool_call(policy.next_tool(req.observation, req.transcript)))
        return out


# --------------------------------------------------------------------------- #
# Completion mask: transcript -> (prompt_ids, completion_ids, completion_mask)
# --------------------------------------------------------------------------- #
def _as_token_ids(out: Any, tokenizer: Any) -> list[int]:
    """Normalize an ``apply_chat_template`` result to a flat ``list[int]``.

    Real tokenizers vary: some return the templated string, some a
    ``BatchEncoding``/dict, some a tensor, some a ``[[...]]`` batch.
    """
    if isinstance(out, str):
        out = tokenizer.encode(out, add_special_tokens=False)
    else:
        # BatchEncoding is not a dict subclass — iterating it yields its keys, so
        # isinstance(out, dict) misses it.
        ids = getattr(out, "input_ids", None)
        if ids is None and hasattr(out, "keys") and "input_ids" in out:
            ids = out["input_ids"]
        if ids is not None:
            out = ids
    if hasattr(out, "tolist"):  # torch/np tensor
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):  # unwrap a batch dim [[...]] -> [...]
        out = out[0]
    return [int(t) for t in out]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest shared leading run of two id sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_masked_completion(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    n_prompt_messages: int,
    enable_thinking: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Tokenize a full transcript and mark only assistant spans as trainable.

    Returns ``(prompt_ids, completion_ids, completion_mask)``: ``completion_mask`` is
    ``1`` on tokens authored by an ``assistant`` turn, ``0`` on tool-result turns and
    template scaffolding — the standard multi-turn-SFT masking GRPO needs.

    Each assistant turn is anchored independently rather than diffed incrementally:
    for turn ``i``, ``context = render(messages[:i], add_generation_prompt=True)`` is
    what the model was conditioned on, and ``with_turn = render(messages[:i+1],
    add_generation_prompt=False)`` is the stored render through that turn. The
    authored span is ``with_turn`` past its common prefix with ``context``; its
    position in the full render is ``context``'s common prefix with ``full_ids``.

    This survives a real Qwen3 quirk: under ``enable_thinking=False`` it attaches an
    empty ``<think></think>`` to the *last* assistant turn of a render but strips it
    from earlier ones, so the stored transcript is not prefix-consistent turn to
    turn — a naive incremental diff breaks on exactly that. Raises ``ValueError`` if
    the authored tokens don't appear verbatim at the anchored position in
    ``full_ids``, since a silently wrong mask would train the wrong tokens.
    """
    if not 0 <= n_prompt_messages <= len(messages):
        raise ValueError(f"n_prompt_messages {n_prompt_messages} out of range for {len(messages)} messages")

    def render(upto: int, add_generation_prompt: bool) -> list[int]:
        # enable_thinking must match how the dataset prompt was baked (render_prompts);
        # fall back to no kwarg for a template that doesn't accept it.
        try:
            out = tokenizer.apply_chat_template(
                messages[:upto],
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            out = tokenizer.apply_chat_template(
                messages[:upto],
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        return _as_token_ids(out, tokenizer)

    full_ids = render(len(messages), add_generation_prompt=False)
    full_mask: list[int] = [0] * len(full_ids)
    for i in range(n_prompt_messages, len(messages)):
        if messages[i].get("role") != "assistant":
            continue
        context = render(i, add_generation_prompt=True)
        with_turn = render(i + 1, add_generation_prompt=False)
        authored = with_turn[_common_prefix_len(context, with_turn) :]
        start = _common_prefix_len(context, full_ids)
        end = start + len(authored)
        if full_ids[start:end] != authored:
            raise ValueError(
                "assistant turn does not align to the whole-conversation render; cannot "
                "build a reliable token-level completion mask"
            )
        for t in range(start, end):
            full_mask[t] = 1

    # Prompt = the model's turn-0 context; split the completion where it diverges
    # from the full render.
    prompt_ids = render(n_prompt_messages, add_generation_prompt=True)
    split = _common_prefix_len(prompt_ids, full_ids)
    return prompt_ids, full_ids[split:], full_mask[split:]


def to_grpo_example(
    rollout: MultiTurnRollout,
    tokenizer: Any,
    *,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """Assemble the ``(prompt, completion, mask, reward)`` GRPO training example.

    ``reward`` is the trusted whole-rollout reward; GRPO's group-relative advantage
    is computed across a group of these. ``enable_thinking`` must match the value
    used to bake the dataset prompt so the mask's token boundaries line up.
    """
    prompt_ids, completion_ids, completion_mask = build_masked_completion(
        rollout.messages,
        tokenizer,
        n_prompt_messages=len(rollout.prompt_messages),
        enable_thinking=enable_thinking,
    )
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "reward": float(rollout.reward),
    }


__all__ = [
    "TOOL_ROLE",
    "BatchGenerator",
    "BatchToolPolicyGenerator",
    "CompletionGenerator",
    "MultiTurnRollout",
    "ToolPolicyGenerator",
    "TurnRequest",
    "build_masked_completion",
    "render_tool_call",
    "render_tool_result",
    "run_batched_training_episodes",
    "run_batched_training_episodes_local",
    "run_training_episode",
    "run_training_episode_local",
    "to_grpo_example",
]
