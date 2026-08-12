"""Trusted per-family success predicates (SPEC §7/§9).

Each predicate answers a single task family's target condition by reading **only
trusted simulator state** — the disturbance engine's committed ``flips`` and its
decoded ``flipped_row_keys``, plus a trusted byte reader backed by the worker's
functional memory — never policy-provided logs, stdout, or claims (SPEC §9). The
compiler (`rowhammer_env/tasks/compiler.py`) hands each predicate its
`CompiledTask`, which carries the sampled target for the episode. Splitting one
predicate per family (replacing the monolithic ``_trusted_success``) keeps each
family's win condition independently testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeAlias

if TYPE_CHECKING:  # avoid a runtime import cycle with the engine/compiler
    from ..disturbance import DisturbanceEngine
    from ..tasks.compiler import CompiledTask


TrustedByteReader: TypeAlias = Callable[[int], int]
SuccessPredicate: TypeAlias = Callable[["CompiledTask", "DisturbanceEngine", TrustedByteReader], bool]


# @spec:rl-reward @spec:invariant-trusted-reward
def _target_bankrow_flip(
    task: "CompiledTask", dist: "DisturbanceEngine", read_byte: TrustedByteReader
) -> bool:
    """Any cell in the target row has flipped (SPEC §7 families 1-3, 5, 8, 9).

    "The target row" is a full physical coordinate, so this compares the trusted
    *decoded* victim key — ``(channel, rank, bankgroup, bank, row)``, recorded by
    the engine from the real issued events — and not ``addr // row_bytes``. The
    row *index* alone names one physical row per bank (16 of them on the admitted
    DDR4 geometry), so an index comparison credits a flip in a different physical
    row; and under a per-episode-secret row->bank mapper (P24) the flip's linear
    address does not encode its row at all. ``CompiledTask.target_row_key`` is
    fixed at compile time from the same decode the worker performs.
    """
    del read_byte
    return task.target_row_key in dist.flipped_row_keys


def _any_flip(task: "CompiledTask", dist: "DisturbanceEngine", read_byte: TrustedByteReader) -> bool:
    """Any simulated cell anywhere has flipped (SPEC §7 family 4)."""
    del task, read_byte
    return bool(dist.flips)


# @spec:rl-reward @spec:invariant-trusted-reward
def _target_cell_flip(
    task: "CompiledTask", dist: "DisturbanceEngine", read_byte: TrustedByteReader
) -> bool:
    """The disturbed target bit reads the requested value (SPEC §7 family 6)."""
    if dist.flips.get(task.target_addr) != task.target_bit:
        return False
    bit_mask = 1 << task.target_bit
    byte = read_byte(task.target_addr)
    return (byte & bit_mask) == (task.target_value & bit_mask)


# @spec:rl-reward @spec:invariant-trusted-reward
def _pattern_target(
    task: "CompiledTask", dist: "DisturbanceEngine", read_byte: TrustedByteReader
) -> bool:
    """The disturbed target byte reads the desired masked value (SPEC §7 family 7)."""
    if task.target_addr not in dist.flips:
        return False
    byte = read_byte(task.target_addr)
    return (byte & task.target_mask) == (task.target_value & task.target_mask)


# @spec:task-families
# One predicate per SPEC §7 family. ``mitigation_aware`` shares the target-row
# condition (family 8 is "same objective with a mitigation enabled");
# ``profile_generalization`` shares the any-flip condition on the eval split.
PREDICATES: dict[str, SuccessPredicate] = {
    "known_target_anybit": _target_bankrow_flip,
    "target_row": _target_bankrow_flip,
    "hidden_target": _target_bankrow_flip,
    "unknown_adjacency": _target_bankrow_flip,
    "bounded_sweep": _target_bankrow_flip,
    "hidden_adjacency": _target_bankrow_flip,
    "mitigation_aware": _target_bankrow_flip,
    "low_disclosure": _target_bankrow_flip,
    "any_flip": _any_flip,
    "profile_generalization": _any_flip,
    "target_cell": _target_cell_flip,
    "pattern_target": _pattern_target,
}


def success_for(
    task: "CompiledTask | None",
    dist: "DisturbanceEngine | None",
    read_byte: TrustedByteReader,
) -> bool:
    """Evaluate the compiled task's family predicate against trusted state."""
    if task is None or dist is None:
        return False
    predicate = PREDICATES.get(task.family)
    if predicate is None:
        raise KeyError(f"no success predicate for family {task.family!r}")
    return predicate(task, dist, read_byte)
