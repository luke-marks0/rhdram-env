"""Deterministic splits and held-out statistical validation."""

from __future__ import annotations

from .split import (
    HOLDOUT_CHIPS,
    Split,
    is_holdout_row,
    partition_rd,
    split_summary,
)

__all__ = [
    "HOLDOUT_CHIPS",
    "Split",
    "is_holdout_row",
    "partition_rd",
    "split_summary",
]
