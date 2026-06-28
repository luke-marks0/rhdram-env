"""Deterministic canonical JSON.

Fitted floats are rounded to a fixed precision and serialized with sorted keys and
no insignificant whitespace, so the same pinned source and recipe always produce
byte-identical profile bytes (and therefore a byte-identical signature). NaN/Inf are
rejected rather than emitted, so a non-finite fit fails closed.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

ROUND_DECIMALS = 6


def canonicalize(obj: Any) -> Any:
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in profile payload: {value!r}")
        return round(value, ROUND_DECIMALS)
    if isinstance(obj, str) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [canonicalize(v) for v in obj]
    raise TypeError(f"cannot canonicalize {type(obj).__name__}")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
