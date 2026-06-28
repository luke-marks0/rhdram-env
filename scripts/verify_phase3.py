#!/usr/bin/env python3
"""Phase 3 admission gate: empirical DDR4 read-disturbance profile.

Runs the Phase 0 policy gate, then checks the Phase 3 admission criteria:

- the ``ddr4_vts25`` source is admitted with resolved provenance, and the on-disk
  data matches the committed per-file hashes;
- the committed profile package exists and its signature verifies against the trust
  store (tampering would raise ``PROFILE_REJECTED``);
- the package reports held-out validation as passed;
- rebuilding from the pinned source reproduces the committed package byte-for-byte.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_builder.errors import ProfileError  # noqa: E402
from profile_builder.manifest import load_source_pin  # noqa: E402
from profile_builder.package.build import build, verify_package  # noqa: E402
from profile_builder.paths import OUT_DIR  # noqa: E402


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase0.py"], cwd=ROOT, check=True)

    pin = load_source_pin(verify_data=True)
    if pin.source_id != "ddr4_vts25":
        raise SystemExit("unexpected source pin")

    profile = verify_package()
    if not profile["validation"]["passed"]:
        raise SystemExit("committed profile reports failed held-out validation")

    committed = (OUT_DIR / "profile.json").read_bytes()
    rebuilt = build(write=False)
    if rebuilt["payload_sha256"] != hashlib.sha256(committed).hexdigest():
        raise SystemExit("rebuild does not reproduce the committed profile.json")

    print(f"phase3 verification passed ({profile['validation']['n_gated']} held-out checks)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileError as exc:
        raise SystemExit(f"{exc.code}: {exc.message}")
