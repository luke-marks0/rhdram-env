"""Canonical repository paths for the profile pipeline."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Vendored / fetched upstream source (git-ignored; populated by
# scripts/fetch_phase3_sources.py or an offline vendor of the pinned commit).
SOURCE_DIR = ROOT / "third_party/ReadDisturbanceVTS25"
DATA_DIR = SOURCE_DIR / "data"

# Committed provenance: the authoritative per-file hash list for the source data.
SOURCE_HASHES = ROOT / "profile_builder/sources/ddr4_vts25.sha256sums"

# Committed build recipe and trust store.
CONFIG = ROOT / "configs/profiles/ddr4_vts25_v1.yaml"
TRUST_DIR = ROOT / "profile_builder/trust"
SIGNING_SEED = TRUST_DIR / "dev_signing_seed.txt"

# Committed output package.
OUT_DIR = ROOT / "profiles/ddr4_vts25_v1"

MANIFEST = ROOT / "SOURCE_MANIFEST.yaml"


def require_source_data() -> pathlib.Path:
    """Return the source data dir, failing closed if it is absent."""
    if not DATA_DIR.is_dir():
        raise SystemExit(
            "missing source data: "
            f"{DATA_DIR.relative_to(ROOT)}; run scripts/fetch_phase3_sources.py"
        )
    return DATA_DIR
