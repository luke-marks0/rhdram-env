"""Multi-turn training rollout + completion-mask assembly (P27).

The *evaluation* path (``rollout.run_episode`` / ``run_episode_local``) is already
turn-by-turn interactive. The **GRPO training** path, by contrast, is single-shot:
``grpo_env.evaluate_item`` parses one completion into an action list and replays it.
P27's lift is training-side — give the trainer genuine multi-turn trajectories and
feed them in with correct loss masking.

This module owns the two host-runnable, ``torch``/``trl``-free pieces:

1. **The multi-turn trajectory loop** (:func:`run_training_episode_local` and its
   async HTTP twin :func:`run_training_episode`): render the running chat transcript
   (system + task prompt, then alternating ``assistant`` tool-call turns and
   ``tool``-result turns built from the real :class:`Phase2Observation`), call a
   :class:`CompletionGenerator` for one short completion per turn, parse it through the
   *same* path the single-shot trainer uses (:func:`grpo_env.parse_actions` /
   ``_coerce_action``), ``step()`` the environment, and repeat until the model emits
   ``episode.finish``, the budget is exhausted, or a ``max_turns`` cap is hit. Reward
   attribution is unchanged from the single-shot path: the whole trajectory earns the
   *final* trusted episode reward (``0.0``/``1.0`` from ``_trusted_success()``); GRPO's
   group-relative advantage is per full rollout, never per turn.

2. **The completion mask** (:func:`build_masked_completion`): split a full transcript
   into ``prompt_ids`` + ``completion_ids`` + a token-level ``completion_mask`` that is
   ``1`` exactly on assistant-authored spans and ``0`` on the interleaved tool-result
   turns and all template scaffolding — the multi-turn SFT masking pattern GRPO needs
   so only model-authored tokens enter the loss. It depends only on the tokenizer's
   ``apply_chat_template`` (the real ``transformers`` interface), so it stays
   ``torch``-free and unit-testable on the host with a fake tokenizer.

The actual ``trl``/``torch`` trainer wiring stays in ``scripts/train_grpo.py`` (never
in an importable library module), so importing this for a host-side unit test needs
neither.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from rowhammer_env import Phase2Action
from rowhammer_env.observability.metrics import EpisodeResult, TrajectoryStep

from .grpo_env import build_messages, parse_actions
from .policies import ToolCall, ToolPolicy

# The role used for tool-result turns. Masking keys off ``role == "assistant"`` (see
# :func:`build_masked_completion`), so any non-assistant role works; ``"tool"`` is the
# semantically correct one and is understood by chat templates that expose a tool role
# (Qwen3 does). On a template without one, map it to ``"user"`` at render time — that
# is an instance-side template concern, not a masking-logic one.
TOOL_ROLE = "tool"

# The full-trace disclosure emits a per-activation ``trace_tail`` — often 50+ near-identical
# RD rows (~1-2k tokens) that repeat in EVERY tool turn, bloating each subsequent turn's
# prompt (slower prefill, more memory, the 2k-token budget spent in a turn or two) while
# adding no signal the digest doesn't already carry (``acts_delta`` / ``per_addr_hits`` are
# the discriminators). Cap how many rows are fed back to the model. 0 drops it entirely.
_MAX_TRACE_TAIL = 4


# --------------------------------------------------------------------------- #
# Generator interface: turn context -> one assistant completion (text)
# --------------------------------------------------------------------------- #
class CompletionGenerator(Protocol):
    """Produce one assistant completion for the current turn.

    A real model generator uses ``messages`` (the rendered chat transcript) and
    ignores the structured ``observation``/``transcript``; a :class:`ToolPolicyGenerator`
    does the reverse. Both are supplied so either kind of decision-maker drops in
    without changing the loop.
    """

    def __call__(
        self, messages: list[dict[str, str]], observation: Any, transcript: list[dict[str, Any]]
    ) -> str:
        ...


def render_tool_call(call: ToolCall) -> str:
    """Render a tool call as the fenced-JSON completion the system prompt asks for.

    Round-trips through :func:`grpo_env.parse_actions`: ``parse_actions(render_tool_call
    (c))[0] == c`` for any tool call whose args are JSON values (all the env's are).
    That exact round-trip is what makes a :class:`ToolPolicyGenerator`-driven rollout
    trace-equivalent to driving the same ``ToolPolicy`` through ``run_episode_local``.
    """
    body = json.dumps({"actions": [{"tool": call.name, "args": call.args}]})
    return "```json\n" + body + "\n```"


def _trim_trace_tail(feedback: Any, limit: int = _MAX_TRACE_TAIL) -> Any:
    """Return a copy of ``feedback`` with any ``trace_tail`` list capped to ``limit`` rows.

    Caps a ``trace_tail`` at the feedback top level and inside ``timing_digest`` (the two
    places the full-trace disclosure emits it), keeping only the *last* ``limit`` rows and
    recording the original length as ``trace_tail_len`` so the count isn't lost. Copies the
    dicts it edits — never mutates the observation's own feedback, so the trajectory keeps
    the full trace and only the text fed back to the model is trimmed.
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
    """Render the disclosed step observation as the ``tool``-turn text fed back.

    Only policy-visible fields — the leakage guard has already stripped hidden
    coordinates from ``feedback``/``last_action`` on the ``step()`` path, so this
    carries nothing the observation itself doesn't. The verbose ``trace_tail`` is capped
    (:func:`_trim_trace_tail`) so it doesn't bloat every subsequent turn's prompt.
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

    Each turn it calls ``policy.next_tool(observation, transcript)`` and serializes the
    chosen :class:`ToolCall` to the fenced-JSON completion a trained model would emit,
    so a deterministic reference policy can drive the *training* loop and produce a
    trajectory identical to the eval loop's — the P27 trace-equivalence check.
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

    ``messages`` is the whole chat transcript (prompt + every assistant/tool turn);
    ``prompt_messages`` is the fixed prefix (system + task). ``result``/``reward`` carry
    the trusted episode outcome — the single scalar reward for the entire trajectory.
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

    Uses the exact parse path the single-shot trainer uses; an unparseable turn
    degrades to ``episode.finish`` (an empty parse earns 0 reward, same as today).
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

    Mirrors :func:`rollout.run_episode_local` turn-for-turn — one decision per real
    ``step()`` until ``done``/``max_turns`` — but records the rendered chat transcript
    alongside the trajectory. Driving a :class:`ToolPolicyGenerator` here yields a
    trajectory + reward identical to driving the wrapped ``ToolPolicy`` through
    ``run_episode_local`` (the host analog of the HTTP path; the host has no server).
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
    """Collect a multi-turn trajectory over the HTTP/WS transport (instance path).

    Same loop as :func:`run_training_episode_local`, driving the real OpenEnv server
    via :class:`RowHammerClient`. Kept here (not in ``scripts/``) so the loop is one
    implementation; the ``trl``/``torch`` generation still lives in the training
    entrypoint. Imports the client lazily — the host has no websocket runtime.
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
# Completion mask: transcript -> (prompt_ids, completion_ids, completion_mask)
# --------------------------------------------------------------------------- #
def _as_token_ids(out: Any, tokenizer: Any) -> list[int]:
    """Normalize an ``apply_chat_template`` result to a flat ``list[int]``.

    ``apply_chat_template(tokenize=True)`` is *documented* to return token ids, but real
    tokenizers vary: some return the templated **string** (tokenize effectively ignored),
    some a ``BatchEncoding``/dict, some a tensor, some a ``[[...]]`` batch. The host fake
    tokenizer always returns ``list[int]``, so this only bites against a real model. Force
    a flat int list here (encoding a returned string with ``add_special_tokens=False`` —
    in-text special tokens still map to their ids, so it stays prefix-consistent).
    """
    if isinstance(out, str):
        out = tokenizer.encode(out, add_special_tokens=False)
    else:
        # BatchEncoding is NOT a dict subclass — iterating it yields its *keys*
        # ("input_ids", ...), so isinstance(out, dict) misses it. Pull input_ids via the
        # attribute (BatchEncoding exposes it) or the mapping key (plain dict).
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

    Returns ``(prompt_ids, completion_ids, completion_mask)`` where ``completion_mask``
    is ``1`` on tokens authored by an ``assistant`` turn (its content plus the turn-end
    the model emits) and ``0`` on the interleaved tool-result turns and role scaffolding.
    This is the standard multi-turn-SFT / tool-RL masking so GRPO's loss only covers
    model-authored tokens.

    Method (robust against real chat templates, incl. Qwen3):

    * The training sequence is the canonical whole-conversation render
      (``add_generation_prompt=False``) — ``full_ids``.
    * ``prompt_ids`` is rendered with ``add_generation_prompt=True`` — identical to how
      ``render_prompts`` bakes the dataset prompt TRL matched on, *including* any
      generation-only scaffolding a template injects there (e.g. Qwen3's empty
      ``<think></think>`` under ``enable_thinking=False``). ``completion_ids`` begins where
      the prompt and ``full_ids`` diverge (their longest common prefix), so the turn-0
      assistant *header* lives in the prompt (not duplicated) and its generated *content*
      opens the completion.
    * Each assistant turn's trainable span is attributed **directly**, not by an
      append-only diff of the stored transcript. For turn ``i``,
      ``context = render(messages[:i], add_generation_prompt=True)`` is the exact context
      the model was conditioned on, and
      ``with_turn = render(messages[:i+1], add_generation_prompt=False)`` is the stored
      render through that turn. The authored tokens are ``with_turn`` past its longest
      common prefix with ``context`` (the shared assistant header + any think
      scaffolding), and their position in ``full_ids`` is the longest common prefix of
      ``context`` and ``full_ids``. A verify step confirms the authored tokens appear
      verbatim at that position in ``full_ids`` before anything is masked.

    This never assumes the stored transcript is prefix-consistent turn to turn — which
    Qwen3 is **not**: under ``enable_thinking=False`` it attaches an empty
    ``<think></think>`` to the *last* assistant turn of a render but strips it from
    earlier ones, so appending a later turn rewrites an earlier turn's tokens. The old
    incremental diff fails closed on exactly that; anchoring per turn on the generation
    prompt sidesteps it because the message *before* an assistant turn is always a
    tool/user result (never a rewritten assistant), keeping the context a clean anchor.
    Using the common prefix (not a strict one) also covers the dual case — scaffolding
    the generation prompt injects that the *stored* turn drops.

    Depends only on ``apply_chat_template`` — no ``torch`` — so it unit-tests on the host
    with a fake tokenizer. Still fails closed (``ValueError``) on a genuinely pathological
    template whose authored tokens do not appear verbatim at the anchored position in
    ``full_ids`` (e.g. one that rewrites earlier tokens as the transcript grows), since a
    silently wrong mask would train the wrong tokens.
    """
    if not 0 <= n_prompt_messages <= len(messages):
        raise ValueError(f"n_prompt_messages {n_prompt_messages} out of range for {len(messages)} messages")

    def render(upto: int, add_generation_prompt: bool) -> list[int]:
        # Pass enable_thinking so the prompt render matches how the dataset prompt was
        # baked (render_prompts); a template that doesn't accept the kwarg (the host fake,
        # older templates) falls back to rendering without it.
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
        # Authored tokens = the stored render past the header/scaffolding it shares with
        # the generation prompt; their anchor = where that context diverges from full_ids.
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

    # Prompt = the model's turn-0 context (matches the baked dataset prompt); split the
    # completion at where the prompt and the full render first diverge.
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

    The trusted trajectory reward is the single scalar for the whole rollout — GRPO's
    group-relative advantage is computed across a group of these, per full rollout.
    ``enable_thinking`` must match the value used to bake the dataset prompt so the
    prompt/mask token boundaries line up.
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
    "CompletionGenerator",
    "MultiTurnRollout",
    "ToolPolicyGenerator",
    "build_masked_completion",
    "render_tool_call",
    "render_tool_result",
    "run_training_episode",
    "run_training_episode_local",
    "to_grpo_example",
]
