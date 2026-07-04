"""Per-standard read-disturbance adapter for the disturbance engine (P15).

Turns the standard-level facts in :mod:`profile_builder.standards` into the
concrete, geometry-derived parameters the :class:`DisturbanceEngine` consumes:
the physical blast neighbourhood, the refresh-window length, and the Ramulator
``dram.impl``/controller a profile of this standard runs on. This is what makes
the engine *standard-generic* (fixing defect F): nothing DDR4-specific is baked
into the engine — every standard-dependent value is looked up per standard and
derived from the real ``DRAMSpec`` geometry the worker publishes.

The statistical flip model still comes only from an admitted empirical profile of
the *matching* standard; there is no parameter pooling across standards.
"""

from __future__ import annotations

from dataclasses import dataclass

from profile_builder.standards import (
    STANDARD_FACTS,
    SUPPORTED_STANDARDS,
    StandardFacts,
    UnsupportedStandard,
    facts_for,
)

from .geometry import Geometry

__all__ = [
    "STANDARD_FACTS",
    "SUPPORTED_STANDARDS",
    "StandardFacts",
    "StandardModel",
    "UnsupportedStandard",
    "facts_for",
    "ramulator_impl_for",
]


# Coupling weight of the ±2 "half-double" ring, relative to the immediate
# neighbour, for profiles that characterise it. Half-double next-nearest rows
# couple far more weakly than the classic ±1 neighbour (SPEC §5.3).
HALF_DOUBLE_WEIGHT = 0.5


# Ramulator ``dram.impl`` + controller to instantiate per standard (P15 task 3:
# "select the Ramulator dram.impl + controller per profile standard"). All
# current standards use the generic DDR controller; the RFM/VRR variants differ
# only in the DRAM model. This is the simulator-side wiring, so it lives in the
# engine package rather than in the source-traced facts table.
RAMULATOR_IMPL: dict[str, tuple[str, str]] = {
    "DDR4": ("DDR4", "GenericDDR"),
    "DDR4_VRR": ("DDR4_VRR", "GenericDDR"),
    "DDR5": ("DDR5", "GenericDDR"),
    "DDR5_RFM": ("DDR5_RFM", "GenericDDR"),
    "DDR5_VRR": ("DDR5_VRR", "GenericDDR"),
    "DDR5_RFM_VRR": ("DDR5_RFM_VRR", "GenericDDR"),
    "HBM2": ("HBM2", "GenericDDR"),
}


def ramulator_impl_for(standard: str) -> tuple[str, str]:
    """Return the ``(dram_impl, controller)`` names for ``standard`` or fail closed."""
    try:
        return RAMULATOR_IMPL[standard]
    except KeyError:
        raise UnsupportedStandard(f"UNAVAILABLE_CAPABILITY:standard {standard}")


@dataclass(frozen=True)
class StandardModel:
    """Geometry-bound, standard-specific parameters for the disturbance engine.

    Built from the real ``DRAMSpec`` geometry the worker reports plus the
    standard's :class:`StandardFacts`. Everything the engine needs that used to be
    a DDR4 constant — the blast set, the refresh window, the linear row stride —
    is derived here, so the *same* engine drives any admitted standard.
    """

    facts: StandardFacts
    geometry: Geometry
    blast: tuple[tuple[int, float], ...]
    refresh_window: int

    @property
    def standard(self) -> str:
        return self.facts.standard

    @property
    def row_bytes(self) -> int:
        """Linear stride between two physically adjacent rows (real geometry)."""
        return self.geometry.row_stride

    @property
    def tx_bytes(self) -> int:
        return self.geometry.tx_bytes

    def ramulator_impl(self) -> tuple[str, str]:
        return ramulator_impl_for(self.facts.standard)

    @classmethod
    def from_geometry(cls, geometry: Geometry, *, half_double: bool = False) -> "StandardModel":
        """Derive the standard model from a real geometry, failing closed.

        The geometry's ``standard`` must have an admitted adapter, and its level
        structure must match the standard's dimensions (e.g. HBM must expose a
        ``PseudoChannel`` level, DDR must not) — this is what guarantees no DDR4
        assumption leaks into a non-DDR standard.
        """
        facts = facts_for(geometry.standard)
        _validate_geometry(facts, geometry)

        blast = facts.blast_neighbors
        if half_double and 2 not in {d for d, _ in blast}:
            blast = blast + ((2, HALF_DOUBLE_WEIGHT),)
        return cls(
            facts=facts,
            geometry=geometry,
            blast=tuple(blast),
            refresh_window=facts.refresh_commands_per_window,
        )


def _validate_geometry(facts: StandardFacts, geometry: Geometry) -> None:
    """Fail closed if the geometry's levels contradict the standard's dimensions."""
    has_pseudo = "pseudochannel" in geometry.level_names
    if facts.pseudo_channel != has_pseudo:
        raise UnsupportedStandard(
            f"PROFILE_REJECTED:{facts.standard} pseudo_channel={facts.pseudo_channel} "
            f"but geometry {'has' if has_pseudo else 'lacks'} a PseudoChannel level"
        )
    if "row" not in geometry.level_names:
        raise UnsupportedStandard(f"PROFILE_REJECTED:{facts.standard} geometry has no Row level")
