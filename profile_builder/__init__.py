"""Phase 3 empirical profile pipeline.

Ingests the admitted DDR4 read-disturbance source (``ddr4_vts25``), canonicalizes
the real-chip measurements, fits an empirical read-disturbance model, validates it
on held-out units, and packages a signed profile.

Every step fits from real measurements only. There is no fabricated calibration
fallback: a missing or hash-mismatched source fails closed instead of substituting
stand-in data.
"""

from __future__ import annotations

PROFILE_ID = "ddr4_vts25_v1"
SOURCE_ID = "ddr4_vts25"

__all__ = ["PROFILE_ID", "SOURCE_ID"]
