"""Deterministic train / held-out split.

Held-out units are reserved two ways, and the fit never sees either:

1. Held-out chips (module generalization): one chip per multi-chip family is
   reserved entirely. The single-chip family ``axmicr`` is train-only and is
   covered by row-level holdout instead.
2. Held-out rows (within-chip generalization): a deterministic ~25% of victim
   rows in every training chip, selected by hashing ``"<chip>:<row>"``.

The selection depends only on the chip id and victim row, never on measured
values, so it is stable and independent of the fit (TEST_PLAN P3).
"""

from __future__ import annotations

import dataclasses
import hashlib

from ..ingest import RdRecord

# One held-out chip per multi-chip family (hisasa, hyhy, sasa). Documented and fixed.
HOLDOUT_CHIPS: frozenset[str] = frozenset({"hisasa03", "hyhy1e", "sasa29"})

# Reserve roughly one quarter of victim rows from training chips for validation.
_ROW_HOLDOUT_MODULUS = 4


@dataclasses.dataclass(frozen=True)
class Split:
    train: list[RdRecord]
    holdout: list[RdRecord]


def is_holdout_row(chip: str, victim_row: int) -> bool:
    digest = hashlib.sha256(f"{chip}:{victim_row}".encode()).digest()
    return digest[0] % _ROW_HOLDOUT_MODULUS == 0


def _is_holdout(r: RdRecord) -> bool:
    return r.chip in HOLDOUT_CHIPS or is_holdout_row(r.chip, r.victim_row)


def partition_rd(records: list[RdRecord]) -> Split:
    train: list[RdRecord] = []
    holdout: list[RdRecord] = []
    for r in records:
        (holdout if _is_holdout(r) else train).append(r)
    return Split(train=train, holdout=holdout)


def split_summary(records: list[RdRecord]) -> dict[str, object]:
    split = partition_rd(records)
    train_chips = sorted({r.chip for r in split.train})
    holdout_chips = sorted({r.chip for r in split.holdout})
    return {
        "holdout_chips": sorted(HOLDOUT_CHIPS),
        "row_holdout_fraction_nominal": 1.0 / _ROW_HOLDOUT_MODULUS,
        "n_train_records": len(split.train),
        "n_holdout_records": len(split.holdout),
        "train_chips": train_chips,
        "holdout_chips_observed": holdout_chips,
        "holdout_units": "chip-level + hashed 25% of victim rows",
    }
