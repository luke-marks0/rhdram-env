"""Multi-turn GRPO: normalize rewards within a group into advantages, then a
PPO-clipped policy update with a KL penalty to the frozen base model.

Multi-turn handling is deliberately simple: every assistant turn of an episode is
one training sequence (prompt-so-far + that turn's sampled tokens), and the whole
episode's group-normalized advantage is broadcast to all of its turns. Sequences are
processed one at a time with gradient accumulation — no padding, no batching gymnastics
— which is plenty for a single-GPU PoC and keeps the loss easy to verify.

``group_advantages`` is pure Python so the advantage math (including the zero-variance
diagnostic the training scope asks for) is unit-testable without torch. Everything
below :func:`grpo_update` needs the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import EpisodeResult

ADV_EPSILON = 1e-6


@dataclass
class TurnSample:
    prompt_ids: list[int]
    gen_ids: list[int]
    old_logprobs: list[float]
    advantage: float


def group_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantages: ``(r - mean) / (std + eps)``.

    Returns all-zeros when the group has no reward variance — GRPO has no learning
    signal there, and the trainer treats that as the diagnostic condition the scope
    calls out (skip the update; stop the run if it persists).
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < ADV_EPSILON:
        return [0.0] * n
    return [(r - mean) / (std + ADV_EPSILON) for r in rewards]


def reward_variance(rewards: list[float]) -> float:
    n = len(rewards)
    if n == 0:
        return 0.0
    mean = sum(rewards) / n
    return sum((r - mean) ** 2 for r in rewards) / n


def samples_from_group(episodes: list[EpisodeResult], advantages: list[float]) -> list[TurnSample]:
    """Flatten a group of episodes into per-turn training sequences.

    Only turns that carry sampled tokens (model rollouts) become samples; scripted
    rollouts contribute none. The episode's advantage is shared by all its turns.
    """
    samples: list[TurnSample] = []
    for episode, advantage in zip(episodes, advantages):
        for turn in episode.turns:
            if turn.gen_ids:
                samples.append(TurnSample(turn.prompt_ids, turn.gen_ids, turn.gen_logprobs, advantage))
    return samples


def grpo_update(policy: Any, samples: list[TurnSample], config: dict[str, Any], optimizer: Any) -> dict[str, float]:
    """One GRPO update over ``samples``: ``updates_per_group`` PPO epochs, returning
    loss/KL/ratio diagnostics. No-op (zero metrics) if there is nothing to train on."""
    import torch

    grpo_cfg = config["grpo"]
    clip_epsilon = float(grpo_cfg["clip_epsilon"])
    kl_beta = float(grpo_cfg["kl_beta"])
    epochs = int(grpo_cfg["updates_per_group"])
    max_grad_norm = float(config["optimizer"]["max_grad_norm"])
    if not samples:
        return {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "ratio": 1.0, "clip_frac": 0.0, "samples": 0}

    # The KL anchor (base model, adapter disabled) is constant across epochs and
    # updates, so score it once per sample instead of every epoch.
    policy.model.train()
    ref_logprobs = [
        policy.logprobs_for(s.prompt_ids, s.gen_ids, grad=False, reference=True).detach() for s in samples
    ]

    metrics = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "ratio": 0.0, "clip_frac": 0.0}
    counted = 0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for sample, ref_lp in zip(samples, ref_logprobs):
            old_lp = torch.tensor(sample.old_logprobs, device=ref_lp.device)
            new_lp = policy.logprobs_for(sample.prompt_ids, sample.gen_ids, grad=True)
            adv = float(sample.advantage)

            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            # k3 KL estimator (always >= 0, low variance): E[exp(r-n) - (r-n) - 1].
            delta = ref_lp - new_lp
            kl = (torch.exp(delta) - delta - 1).mean()
            loss = (policy_loss + kl_beta * kl) / len(samples)
            loss.backward()

            counted += 1
            metrics["loss"] += float(loss) * len(samples)
            metrics["policy_loss"] += float(policy_loss)
            metrics["kl"] += float(kl)
            metrics["ratio"] += float(ratio.mean())
            metrics["clip_frac"] += float((torch.abs(ratio - 1.0) > clip_epsilon).float().mean())

        torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), max_grad_norm)
        optimizer.step()

    for key in metrics:
        metrics[key] /= max(1, counted)
    metrics["samples"] = len(samples)
    return metrics
