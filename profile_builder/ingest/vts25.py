"""Parsers for the four vendored VTS25 read-disturbance measurement types.

Upstream CSV schemas (at the pinned commit):

``*_rd_hcf.csv`` / ``*_rd_ber.csv`` / ``*_rd_rp.csv``
    ``Vic Row, Data Pattern, HC, Aggr. Type, Num. Bitflips, Itr``
``*_ds_ber_sweep.csv``
    ``Temp, Vic Row, HC, Itr``

Semantics:
- ``Aggr. Type`` ``Upper``/``Lower`` are single-sided; ``Double`` is double-sided.
- ``Data Pattern`` ``0xFFFFFFFF`` is all-ones (observed flips are 1->0); ``0x00000000``
  is all-zeros (observed flips are 0->1).
- ``HC`` is the per-measurement activation count (for ``rd_hcf`` it is the count at
  the first observed victim bitflip, i.e. HC-first).
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib

from ..errors import ProfileBuildError
from ..paths import require_source_data

# Data Pattern -> (canonical pattern name, observed bitflip direction).
PATTERNS: dict[str, tuple[str, str]] = {
    "0xFFFFFFFF": ("all_ones", "1->0"),
    "0x00000000": ("all_zeros", "0->1"),
}

# Aggressor type -> exposure class.
AGGR_CLASS: dict[str, str] = {"Upper": "single", "Lower": "single", "Double": "double"}

_RD_KINDS = ("rd_hcf", "rd_ber", "rd_rp")


@dataclasses.dataclass(frozen=True)
class RdRecord:
    chip: str
    family: str
    kind: str  # rd_hcf | rd_ber | rd_rp
    victim_row: int
    pattern: str  # all_ones | all_zeros
    direction: str  # 1->0 | 0->1
    aggr_type: str  # Upper | Lower | Double
    aggr_class: str  # single | double
    hc: int
    num_bitflips: int
    itr: int


@dataclasses.dataclass(frozen=True)
class DsBerRecord:
    chip: str
    family: str
    temp: int
    victim_row: int
    hc: int
    itr: int


def family_of(chip: str) -> str:
    """Family key = leading alphabetic prefix of the upstream module label.

    This is a grouping key from the filename only; it does not assert a DRAM vendor.
    """
    i = 0
    while i < len(chip) and chip[i].isalpha():
        i += 1
    if i == 0:
        raise ProfileBuildError(f"cannot derive family from chip id {chip!r}")
    return chip[:i]


def chip_ids() -> list[str]:
    data = require_source_data()
    ids = sorted(p.name[: -len("_rd_hcf.csv")] for p in data.glob("*_rd_hcf.csv"))
    if not ids:
        raise ProfileBuildError("no rd_hcf source files found")
    return ids


def _read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_rd(kind: str, chips: list[str] | None = None) -> list[RdRecord]:
    """Load one of the RowHammer/RowPress measurement types across chips."""
    if kind not in _RD_KINDS:
        raise ProfileBuildError(f"unknown rd kind {kind!r}")
    data = require_source_data()
    chips = chips if chips is not None else chip_ids()
    out: list[RdRecord] = []
    for chip in chips:
        family = family_of(chip)
        for row in _read_rows(data / f"{chip}_{kind}.csv"):
            pattern = row["Data Pattern"]
            if pattern not in PATTERNS:
                raise ProfileBuildError(f"unexpected data pattern {pattern!r} in {chip} {kind}")
            aggr_type = row["Aggr. Type"]
            if aggr_type not in AGGR_CLASS:
                raise ProfileBuildError(f"unexpected aggressor type {aggr_type!r} in {chip} {kind}")
            name, direction = PATTERNS[pattern]
            out.append(
                RdRecord(
                    chip=chip,
                    family=family,
                    kind=kind,
                    victim_row=int(row["Vic Row"]),
                    pattern=name,
                    direction=direction,
                    aggr_type=aggr_type,
                    aggr_class=AGGR_CLASS[aggr_type],
                    hc=int(row["HC"]),
                    num_bitflips=int(row["Num. Bitflips"]),
                    itr=int(row["Itr"]),
                )
            )
    return out


def load_ds_ber(chips: list[str] | None = None) -> list[DsBerRecord]:
    """Load the temperature-annotated double-sided HC-first sweep across chips."""
    data = require_source_data()
    chips = chips if chips is not None else chip_ids()
    out: list[DsBerRecord] = []
    for chip in chips:
        family = family_of(chip)
        for row in _read_rows(data / f"{chip}_ds_ber_sweep.csv"):
            out.append(
                DsBerRecord(
                    chip=chip,
                    family=family,
                    temp=int(row["Temp"]),
                    victim_row=int(row["Vic Row"]),
                    hc=int(row["HC"]),
                    itr=int(row["Itr"]),
                )
            )
    return out
