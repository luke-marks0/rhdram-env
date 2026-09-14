"""The one multi-turn rollout driver, shared by training and evaluation.

A ``Policy`` decides the next tool call given the structured observation (scripted
policies) or the running chat messages (the model). The driver owns everything else
— prompt building, ``env.step``, tool-result rendering, token bookkeeping, probe
shaping — so a scripted policy and the model produce byte-identical environment
trajectories from the same actions (the "rollout trace equivalence" the training
scope's definition of done calls for). Torch-free: the model policy plugs in as a
``Policy`` and supplies token records, but the loop itself never imports torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rowhammer_env.phase2_env import Phase2Action

from . import prompt
from .metrics import EpisodeResult, TurnRecord
from .probe import count_decisive_probes, probe_shaping, unique_probe_candidates


@dataclass
class GenRecord:
    """Token-level record of one model turn, for the GRPO update. Empty for scripted
    policies. ``prompt_ids`` is the tokenized chat context up to the generation
    prompt; ``gen_ids`` are the exact sampled assistant tokens (the only ones trained
    on); ``gen_logprobs`` are their logprobs at sampling time (the PPO "old" policy).
    """

    prompt_ids: list[int] = field(default_factory=list)
    gen_ids: list[int] = field(default_factory=list)
    gen_logprobs: list[float] = field(default_factory=list)


@dataclass
class ActResult:
    action: dict[str, Any]        # {"tool": str, "args": dict}
    text: str | None = None       # rendered completion; defaults to render_action(action)
    gen: GenRecord | None = None


class Policy(Protocol):
    def act(self, *, obs: Any, messages: list[dict[str, str]]) -> ActResult: ...


def run_episode(
    env: Any,
    *,
    task: dict[str, Any],
    seed: int,
    stage: str,
    policy: Policy,
    max_turns: int,
    shaping_weight: float = 0.0,
    trace_tail: int = prompt.DEFAULT_TRACE_TAIL,
    sft_capture: list[dict[str, Any]] | None = None,
) -> EpisodeResult:
    """Run one episode of ``task`` at ``seed`` under ``policy`` and score it.

    ``shaping_weight`` > 0 adds the bounded probe-shaping term (training only); pass
    0 for the trusted-sparse-reward scoring evaluation and benchmarking use.

    ``sft_capture``, if given, receives one ``{"prompt": messages, "assistant": text}``
    record per turn — the exact context and completion the policy produced — so the
    reference solver can seed a warm-start SFT dataset through this same driver.
    """
    obs = env.reset(seed=seed, task=task)
    if obs.error:
        return EpisodeResult(
            task=task.get("id") or stage, stage=stage, seed=seed, success=0.0, shaping=0.0,
            turns=[], decisive_probes=0, unique_probes=0, valid_calls=0, total_calls=0,
            budget_used={}, error=obs.error.get("code"), done_reason="reset_error",
        )

    metadata = dict(obs.metadata)
    if hasattr(policy, "begin_episode"):
        policy.begin_episode(obs=obs, metadata=metadata)

    messages = [prompt.system_message(), prompt.initial_user_message(obs)]
    turns: list[TurnRecord] = []
    success = 0.0
    done_reason = "max_turns"

    while len(turns) < max_turns:
        result = policy.act(obs=obs, messages=messages)
        text = result.text if result.text is not None else prompt.render_action(result.action)
        if sft_capture is not None:
            sft_capture.append({"prompt": [dict(m) for m in messages], "assistant": text})
        messages.append({"role": "assistant", "content": text})

        action = Phase2Action(tool=result.action.get("tool", ""), args=result.action.get("args", {}))
        obs = env.step(action)
        success = max(success, float(obs.reward or 0.0))

        gen = result.gen or GenRecord()
        turns.append(TurnRecord(
            tool=action.tool, args=action.args, text=text,
            accepted=obs.error is None, error=(obs.error or {}).get("code"),
            feedback=dict(obs.feedback), public_counters=dict(obs.public_counters),
            prompt_ids=gen.prompt_ids, gen_ids=gen.gen_ids, gen_logprobs=gen.gen_logprobs,
        ))
        messages.append(prompt.tool_result_message(obs, trace_tail=trace_tail))

        if obs.done:
            if success >= 1.0:
                done_reason = "success"
            elif obs.error:
                done_reason = obs.error.get("code", "error")
            else:
                done_reason = "finish"
            break

    shaping = probe_shaping(turns, metadata) * shaping_weight if shaping_weight else 0.0
    valid_calls = sum(1 for t in turns if t.accepted)
    budget_used = _budget_used(env, obs)
    return EpisodeResult(
        task=task.get("id") or stage, stage=stage, seed=seed,
        success=1.0 if success >= 1.0 else 0.0, shaping=shaping, turns=turns,
        decisive_probes=count_decisive_probes(turns),
        unique_probes=len(unique_probe_candidates(turns, metadata)),
        valid_calls=valid_calls, total_calls=len(turns),
        budget_used=budget_used, error=(obs.error or {}).get("code"), done_reason=done_reason,
    )


def _budget_used(env: Any, obs: Any) -> dict[str, int]:
    initial = getattr(env, "initial_budgets", {}) or {}
    remaining = obs.metadata.get("budget_remaining") or getattr(env, "budget_remaining", {}) or {}
    return {k: int(initial[k]) - int(remaining.get(k, initial[k])) for k in initial}
