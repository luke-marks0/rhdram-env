"""Training curriculum ordering for the discovery policy (P28).

GRPO on the sparse discovery reward is only learnable in practice if the policy meets
the families in increasing difficulty, each gated by proof that it is *solvable at all*
before any GPU time is spent. This module is the single, host-testable source of truth
for that ordering: a small declarative parser over the ``curriculum:`` block of a
training config into ordered :class:`CurriculumStage` objects.

The ordering (IMPLEMENTATION_PLAN_V3 P28.1):

    Tier 0 known-target  →  ``bounded_sweep`` (Tier 2a) easy → medium → hard
                         →  ``hidden_adjacency`` (Tier 2b) easy → medium → hard

**The gate is the reference policy, not this module.** Each stage carries a
``reference_min_success`` — the fraction of episodes the deterministic P26
``ReferenceProbePolicy`` must solve within the stage's calibrated budget before the
stage earns instance time (``tests/test_curriculum.py`` enforces this on the host with
no ``torch``). This module only *defines and orders* the stages; it computes no reward
and drives no training loop. The trainer (``scripts/train_grpo.py``) consumes the
ordered stages to build its dataset; validation consumes them to re-run the reference
policy after any curriculum change (P28.3).
"""

from __future__ import annotations

from dataclasses import dataclass

# The intended tier progression; a curriculum must not reorder tiers (easier before
# harder). Bands within a tier follow the same easy<medium<hard order.
TIER_ORDER = ("tier0", "tier2a", "tier2b")
BAND_ORDER = ("", "easy", "medium", "hard")


@dataclass(frozen=True)
class CurriculumStage:
    """One ordered curriculum stage: a set of task configs trained together.

    ``reference_min_success`` is the P26 gate — the minimum fraction of seeds the
    deterministic reference policy must solve within budget for this stage to be
    admitted to training. ``multi_turn`` records whether the stage needs the
    turn-by-turn rollout (discovery families do; Tier 0 known-target is single-shot
    solvable but runs multi-turn just as well).
    """

    name: str
    tier: str
    band: str
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    multi_turn: bool
    reference_min_success: float


def _parse_seeds(name: str, entry: dict) -> tuple[int, ...]:
    """Resolve a stage's training seeds from ``seeds:`` or the ``seed_range:`` shorthand.

    * ``seeds: [1, 2, 3, ...]`` — an explicit list, or
    * ``seed_range: [start, stop]`` — every integer from ``start`` to ``stop`` **inclusive**
      (so ``[1, 256]`` is 256 seeds), the practical way to ask for hundreds of instances
      without hand-listing them.

    Exactly one of the two must be present. Returns the seeds in the order given (explicit
    list) or ascending (range).
    """
    has_list = "seeds" in entry and entry.get("seeds") is not None
    has_range = "seed_range" in entry and entry.get("seed_range") is not None
    if has_list and has_range:
        raise ValueError(f"curriculum stage {name!r}: give either 'seeds' or 'seed_range', not both")
    if has_range:
        rng = entry.get("seed_range")
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ValueError(
                f"curriculum stage {name!r}: 'seed_range' must be a [start, stop] pair, got {rng!r}"
            )
        try:
            start, stop = int(rng[0]), int(rng[1])
        except (TypeError, ValueError):
            raise ValueError(f"curriculum stage {name!r}: 'seed_range' bounds must be integers, got {rng!r}")
        if stop < start:
            raise ValueError(
                f"curriculum stage {name!r}: 'seed_range' start ({start}) must be <= stop ({stop})"
            )
        return tuple(range(start, stop + 1))  # inclusive of both ends
    seeds_raw = entry.get("seeds") or []
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(
            f"curriculum stage {name!r}: needs a non-empty 'seeds' list or a 'seed_range: [start, stop]'"
        )
    return tuple(int(s) for s in seeds_raw)


def _stage_from_entry(index: int, entry: dict) -> CurriculumStage:
    if not isinstance(entry, dict):
        raise ValueError(f"curriculum stage #{index} must be a mapping, got {type(entry).__name__}")
    name = str(entry.get("name") or f"stage_{index}")
    tier = str(entry.get("tier") or "").strip()
    if tier not in TIER_ORDER:
        raise ValueError(f"curriculum stage {name!r}: 'tier' must be one of {TIER_ORDER}, got {tier!r}")
    band = str(entry.get("band") or "").strip()
    if band not in BAND_ORDER:
        raise ValueError(f"curriculum stage {name!r}: 'band' must be one of {BAND_ORDER}, got {band!r}")
    tasks_raw = entry.get("tasks") or []
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError(f"curriculum stage {name!r}: 'tasks' must be a non-empty list of config paths")
    tasks = tuple(str(t) for t in tasks_raw)
    seeds = _parse_seeds(name, entry)
    ref = entry.get("reference_min_success")
    if ref is None:
        raise ValueError(f"curriculum stage {name!r}: 'reference_min_success' is required (the P26 gate)")
    ref = float(ref)
    if not 0.0 <= ref <= 1.0:
        raise ValueError(f"curriculum stage {name!r}: 'reference_min_success' must be in [0,1], got {ref}")
    return CurriculumStage(
        name=name,
        tier=tier,
        band=band,
        tasks=tasks,
        seeds=seeds,
        multi_turn=bool(entry.get("multi_turn", True)),
        reference_min_success=ref,
    )


def _assert_non_decreasing_difficulty(stages: list[CurriculumStage]) -> None:
    """A curriculum must never present a harder stage before an easier one."""
    prev = (-1, -1)
    for stage in stages:
        rank = (TIER_ORDER.index(stage.tier), BAND_ORDER.index(stage.band))
        if rank < prev:
            raise ValueError(
                f"curriculum is out of order at stage {stage.name!r}: tier/band "
                f"{stage.tier}/{stage.band or '-'} appears after a strictly harder stage"
            )
        prev = rank


def load_curriculum(cfg: dict) -> list[CurriculumStage]:
    """Parse and validate the ordered ``curriculum:`` block of a training config.

    Returns the stages in file order, after checking every stage is well-formed and
    that difficulty is non-decreasing (Tier 0 before 2a before 2b; easy before medium
    before hard). Raises ``ValueError`` on any malformed or out-of-order curriculum —
    a silently misordered curriculum would defeat the whole point of the gate.
    """
    entries = cfg.get("curriculum")
    if not isinstance(entries, list) or not entries:
        raise ValueError("config has no non-empty 'curriculum:' list")
    stages = [_stage_from_entry(i, entry) for i, entry in enumerate(entries)]
    _assert_non_decreasing_difficulty(stages)
    return stages


def curriculum_task_seed_pairs(stages: list[CurriculumStage]) -> list[tuple[str, int]]:
    """Flatten ordered stages into the ordered ``(task_config_path, seed)`` pairs.

    This is the dataset order the trainer materializes: every task in a stage, over
    every seed, before moving to the next (harder) stage.
    """
    pairs: list[tuple[str, int]] = []
    for stage in stages:
        for task in stage.tasks:
            for seed in stage.seeds:
                pairs.append((task, seed))
    return pairs


__all__ = [
    "BAND_ORDER",
    "TIER_ORDER",
    "CurriculumStage",
    "curriculum_task_seed_pairs",
    "load_curriculum",
]
