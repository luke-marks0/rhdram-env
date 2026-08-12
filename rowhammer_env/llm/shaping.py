"""Narrowly-bounded, training-only reward shaping for discovery rollouts (P28).

The discovery families (``bounded_sweep`` / ``hidden_adjacency``) hand out a *sparse*
end-of-trajectory reward: ``1.0`` only when a real flip satisfies the objective,
``0.0`` otherwise (``_trusted_success()``, SPEC §9). That signal is hard to learn from
directly — a policy has to stumble onto the whole probe→narrow→hammer sequence before
it sees any reward at all. This module adds one *auxiliary* term to make that reward
learnable in practice, kept strictly inside the SPEC §9 envelope.

**What the term rewards.** A small bonus for each turn on which the policy issued a
genuine bank-conflict *probe* that produced a **decisive** same-bank/different-bank
reading — i.e. a repeated two-row alternation whose real ``timing_digest`` cleanly
lands in one regime (``acts_delta == 0`` ⇒ different bank; ``acts_delta`` clearly
positive ⇒ same bank). It is computed **only** from the trusted, already-disclosed
``timing_digest`` (real issued ACT events; §0.2), never from hidden state.

**Why it is inside SPEC §9.**

* It is **outcome-neutral**: a clean *different-bank* reading earns exactly the same
  bonus as a clean *same-bank* reading. It rewards *making a decisive measurement*,
  not *which candidate turned out to be the aggressor* — so it leaks no hint about the
  correct answer (that only the trusted flip can confirm).
* It derives **only** from ``timing_digest`` counts the policy already sees at this
  feedback level (``full_trace``); it fabricates no signal.
* It is **bounded in ``[0, 1]``** and applied with a small weight strictly below
  ``success_weight`` (:func:`validate_shaping_weight`), so no amount of probing can
  earn as much as one real, budget-respecting successful flip. Shaping can never *be*
  success.
* It is **training-only**: eval/benchmark scoring uses the trusted episode reward
  alone (:func:`~rowhammer_env.llm.multiturn_rollout.to_grpo_example`'s ``reward``);
  the shaping term is added by the trainer only when ``reward.probe_shaping_weight``
  is set (default ``0.0``), and vanishes from scoring otherwise.

All of this is host-testable with no ``torch`` — it operates on the plain
:class:`~rowhammer_env.observability.metrics.TrajectoryStep` list a rollout produces.
"""

from __future__ import annotations

from typing import Any


# @spec:train-reward-shaping — shaping is opt-in. Keeping the default beside the
# validation and scoring logic gives training code and tests one authoritative
# value instead of inferring it from the presence of a particular config file.
DEFAULT_PROBE_SHAPING_WEIGHT = 0.0

# A decisive probe is a *pairwise* bank-conflict test: exactly the victim row alternated
# against one candidate. Comparing two rows is what the bank-conflict channel measures;
# a wider access is a hammer or a warm-up, not a same/different-bank probe.
PROBE_ROW_COUNT = 2

# A decisive probe needs a *repeated* interleave: each of the two rows read at least this
# many times (``hits`` = the row-hit RD count, always the read count since a policy RD's
# own row_hit is always true, §0.2.1). A one-shot "warm" reads each row once (``hits==1``)
# and opens both rows regardless of bank, so it carries no bank-conflict signal and must
# not count as a measurement.
MIN_READS_PER_ROW = 2

# Bonus per decisive probe. Four clean measurements saturate the term at 1.0; the
# separate config weight (kept < success_weight) scales the whole thing down.
PER_PROBE_BONUS = 0.25


def _timing_digest(step: Any) -> dict[str, Any]:
    feedback = getattr(step, "feedback", None) or {}
    if not isinstance(feedback, dict):
        return {}
    digest = feedback.get("timing_digest") or {}
    return digest if isinstance(digest, dict) else {}


def is_decisive_probe(step: Any) -> bool:
    """Did this trajectory step make a decisive bank-conflict measurement?

    True iff the step is a ``dram.issue`` whose real ``timing_digest`` reflects a
    *pairwise, repeated* alternation — exactly two rows (:data:`PROBE_ROW_COUNT`), each
    read at least :data:`MIN_READS_PER_ROW` times — that read out **cleanly** in one
    regime: ``acts_delta == 0`` (different bank: the two rows stayed open) or
    ``acts_delta >= PROBE_ROW_COUNT`` (same bank: each access forced a fresh ACT). An
    ambiguous tiny ``acts_delta`` is *not* decisive.

    The pairwise + repeated conditions exclude the things that are not bank-conflict
    probes: a one-shot warm-up (``hits == 1`` per row, opens both rows regardless of
    bank), the multi-row confirmation hammer, and a "hammer everything" control — none
    of which is a same/different-bank measurement.

    Deliberately symmetric: same-bank and different-bank both count, so the bonus
    rewards the *quality of the measurement*, not the answer.
    """
    action = getattr(step, "action", None) or {}
    if not isinstance(action, dict) or action.get("tool") != "dram.issue":
        return False
    digest = _timing_digest(step)
    per_addr = digest.get("per_addr_hits") or {}
    if not isinstance(per_addr, dict) or len(per_addr) != PROBE_ROW_COUNT:
        return False  # not a pairwise victim-vs-candidate alternation
    # ``hits`` is the row-hit RD count (== the read count for a policy RD, §0.2.1). A
    # repeated alternation reads each of the two rows more than once; a warm reads each
    # exactly once and cannot separate same-bank from different-bank.
    reads = [int(b.get("hits", 0) or 0) for b in per_addr.values() if isinstance(b, dict)]
    if len(reads) != PROBE_ROW_COUNT or min(reads) < MIN_READS_PER_ROW:
        return False
    acts_delta = int(digest.get("acts_delta", 0) or 0)
    return acts_delta == 0 or acts_delta >= PROBE_ROW_COUNT


def count_decisive_probes(trajectory: list[Any]) -> int:
    """Number of turns in a trajectory that made a decisive bank-conflict measurement."""
    return sum(1 for step in trajectory if is_decisive_probe(step))


def probe_shaping_reward(rollout: Any, *, per_probe: float = PER_PROBE_BONUS) -> float:
    """Auxiliary shaping reward for one rollout, bounded in ``[0.0, 1.0]``.

    ``rollout`` is a :class:`~rowhammer_env.llm.multiturn_rollout.MultiTurnRollout`
    (or anything exposing a ``trajectory`` of
    :class:`~rowhammer_env.observability.metrics.TrajectoryStep`). Returns
    ``min(1.0, per_probe * decisive_probe_count)`` — small, dense, and derived only
    from the trusted timing digest. Returns ``0.0`` when the policy made no decisive
    probe. This value is *never* an episode reward on its own; the trainer adds it with
    a weight strictly below ``success_weight`` (see :func:`validate_shaping_weight`).
    """
    trajectory = getattr(rollout, "trajectory", None)
    if trajectory is None and isinstance(rollout, (list, tuple)):
        trajectory = rollout
    if not trajectory:
        return 0.0
    n = count_decisive_probes(list(trajectory))
    return min(1.0, float(per_probe) * n)


def shaping_weights(reward_config: dict[str, Any] | None) -> tuple[float, float]:
    """Resolve success and probe-shaping weights from an optional reward config.

    Probe shaping is disabled unless a config explicitly supplies a positive
    ``probe_shaping_weight``. Validation remains conditional on enabling shaping,
    matching the trainer: an absent reward block must be a valid, unshaped run.
    """
    config = reward_config or {}
    success_weight = float(config.get("success_weight", 1.0))
    probe_weight = float(config.get("probe_shaping_weight", DEFAULT_PROBE_SHAPING_WEIGHT))
    if probe_weight > 0.0:
        validate_shaping_weight(success_weight, probe_weight)
    return success_weight, probe_weight


def validate_shaping_weight(success_weight: float, probe_shaping_weight: float) -> None:
    """Fail closed if the shaping weight could rival or exceed a real success (SPEC §9).

    ``probe_shaping_reward`` is bounded in ``[0, 1]``, so a shaping weight ``>=
    success_weight`` would let pure probing earn as much as (or more than) an actual
    flip — exactly the "shaping must never push reward above a successful trajectory"
    line SPEC §9 draws. A non-negative weight strictly below ``success_weight`` is the
    only admissible configuration.
    """
    if probe_shaping_weight < 0.0:
        raise ValueError(f"probe_shaping_weight must be non-negative, got {probe_shaping_weight}")
    if probe_shaping_weight >= success_weight:
        raise ValueError(
            f"probe_shaping_weight ({probe_shaping_weight}) must be strictly below "
            f"success_weight ({success_weight}): the bounded shaping term must never be "
            "able to earn as much as one real successful flip (SPEC §9)."
        )


__all__ = [
    "DEFAULT_PROBE_SHAPING_WEIGHT",
    "MIN_READS_PER_ROW",
    "PER_PROBE_BONUS",
    "PROBE_ROW_COUNT",
    "count_decisive_probes",
    "is_decisive_probe",
    "probe_shaping_reward",
    "shaping_weights",
    "validate_shaping_weight",
]
