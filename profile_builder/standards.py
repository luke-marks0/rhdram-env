"""DRAM-standard read-disturbance facts (SPEC §5.3 / §6, IMPLEMENTATION_PLAN_V2 P15).

Each ``StandardFacts`` records the *standard-level* properties a read-disturbance
model needs that are **not** fitted calibration: the physical blast
neighbourhood a RowHammer aggressor couples into, the JEDEC refresh divisor
(all-bank REF commands per retention window), whether the standard defines
Refresh-Management (RFM) / Victim-Row-Refresh (VRR), on-die ECC, and the extra
addressing dimensions some standards add (pseudo-channel, die stacking). These
are public JEDEC/topology facts, so they live in the source-traceable
``profile_builder`` package and are stamped into every profile package (schema
v2). The *statistical* flip model still comes only from an admitted empirical
profile of the matching standard — no parameter is shared across standards.

``profile_builder`` owns these facts because the signed profile package must
declare, per SPEC §10, which standard dimensions it covers. ``rowhammer_env``
builds the geometry-driven :class:`~rowhammer_env.standards.StandardModel` on top
of this table (and adds the Ramulator ``dram.impl``/controller selection).
"""

from __future__ import annotations

from dataclasses import dataclass


class UnsupportedStandard(ValueError):
    """A DRAM standard with no admitted read-disturbance adapter (fail closed)."""


@dataclass(frozen=True)
class StandardFacts:
    """Read-disturbance-relevant facts of one DRAM standard.

    ``blast_neighbors`` is the immediate physical victim topology as
    ``(row_distance, coupling_weight)`` pairs; distance 1 is the classic
    RowHammer neighbour. ``half_double_supported`` records whether *this
    standard's admitted profiles* additionally model the ±2 half-double coupling
    (kept separate from the raw topology so the engine only extends the blast set
    when the profile characterises it). ``refresh_commands_per_window`` is the
    number of all-bank auto-refresh commands that cover every row once (the JEDEC
    refresh divisor). The remaining booleans flag standard-specific dimensions:
    Refresh-Management, Victim-Row-Refresh, on-die ECC (HBM), pseudo-channel
    addressing (HBM), 3D die stacking (HBM), and whether the profile resolves
    sub-array structure.
    """

    standard: str
    family: str  # coarse family, e.g. "DDR" | "HBM"
    blast_neighbors: tuple[tuple[int, float], ...]
    refresh_commands_per_window: int
    rfm_supported: bool
    vrr_supported: bool
    on_die_ecc: bool
    pseudo_channel: bool
    die_stacking: bool
    half_double_supported: bool = False
    subarray_resolved: bool = False


# JEDEC refresh divisor shared by DDR4/DDR5/HBM2: one retention window
# (tREFW) contains this many auto-refresh intervals (tREFI), so this many
# all-bank refreshes cover every row exactly once.
_REFRESH_DIVISOR = 8192

# The immediate-neighbour RowHammer topology every current standard shares. A
# profile that characterises half-double additionally enables the ±2 ring at
# engine build time (see ``StandardModel.from_geometry``).
_NEIGHBOR_1 = ((1, 1.0),)


STANDARD_FACTS: dict[str, StandardFacts] = {
    "DDR4": StandardFacts(
        standard="DDR4", family="DDR", blast_neighbors=_NEIGHBOR_1,
        refresh_commands_per_window=_REFRESH_DIVISOR,
        rfm_supported=False, vrr_supported=False, on_die_ecc=False,
        pseudo_channel=False, die_stacking=False,
    ),
    "DDR4_VRR": StandardFacts(
        standard="DDR4_VRR", family="DDR", blast_neighbors=_NEIGHBOR_1,
        refresh_commands_per_window=_REFRESH_DIVISOR,
        rfm_supported=False, vrr_supported=True, on_die_ecc=False,
        pseudo_channel=False, die_stacking=False,
    ),
    "DDR5": StandardFacts(
        standard="DDR5", family="DDR", blast_neighbors=_NEIGHBOR_1,
        refresh_commands_per_window=_REFRESH_DIVISOR,
        rfm_supported=False, vrr_supported=False, on_die_ecc=False,
        pseudo_channel=False, die_stacking=False,
    ),
    "DDR5_RFM": StandardFacts(
        standard="DDR5_RFM", family="DDR", blast_neighbors=_NEIGHBOR_1,
        refresh_commands_per_window=_REFRESH_DIVISOR,
        rfm_supported=True, vrr_supported=False, on_die_ecc=False,
        pseudo_channel=False, die_stacking=False,
    ),
    "DDR5_VRR": StandardFacts(
        standard="DDR5_VRR", family="DDR", blast_neighbors=_NEIGHBOR_1,
        refresh_commands_per_window=_REFRESH_DIVISOR,
        rfm_supported=False, vrr_supported=True, on_die_ecc=False,
        pseudo_channel=False, die_stacking=False,
    ),
    "DDR5_RFM_VRR": StandardFacts(
        standard="DDR5_RFM_VRR", family="DDR", blast_neighbors=_NEIGHBOR_1,
        refresh_commands_per_window=_REFRESH_DIVISOR,
        rfm_supported=True, vrr_supported=True, on_die_ecc=False,
        pseudo_channel=False, die_stacking=False,
    ),
    "HBM2": StandardFacts(
        standard="HBM2", family="HBM", blast_neighbors=_NEIGHBOR_1,
        refresh_commands_per_window=_REFRESH_DIVISOR,
        rfm_supported=False, vrr_supported=False, on_die_ecc=True,
        pseudo_channel=True, die_stacking=True,
    ),
}

# The set of standards for which a read-disturbance adapter exists. A standard
# not in this set is absent from capability discovery and fails closed.
SUPPORTED_STANDARDS = frozenset(STANDARD_FACTS)


def facts_for(standard: str) -> StandardFacts:
    """Return the adapter facts for ``standard`` or fail closed."""
    try:
        return STANDARD_FACTS[standard]
    except KeyError:
        raise UnsupportedStandard(f"UNAVAILABLE_CAPABILITY:standard {standard}")


def as_profile_blocks(facts: StandardFacts) -> dict:
    """Serialise adapter facts into the schema-v2 profile blocks (SPEC §10).

    Returned keyed by the top-level profile field they populate: ``topology``,
    ``refresh``, and ``standard_dimensions``. Kept here so the profile builder and
    a consistency check share one definition of the wire shape.
    """
    return {
        "topology": {
            "blast_neighbors": [[int(d), float(w)] for d, w in facts.blast_neighbors],
            "half_double_supported": bool(facts.half_double_supported),
        },
        "refresh": {
            "commands_per_window": int(facts.refresh_commands_per_window),
            "rfm_supported": bool(facts.rfm_supported),
            "vrr_supported": bool(facts.vrr_supported),
        },
        "standard_dimensions": {
            "pseudo_channel": bool(facts.pseudo_channel),
            "die_stacking": bool(facts.die_stacking),
            "on_die_ecc": bool(facts.on_die_ecc),
            "subarray_resolved": bool(facts.subarray_resolved),
        },
    }
