#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import RowHammerTaskEnv
from rowhammer_env.profiles import ADMITTED_PROFILES, DEFERRED_PROFILES, load_profile


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase8.py"], cwd=ROOT, check=True)

    profile = load_profile("ddr4_vts25_v1")
    if profile["standard"] != "DDR4" or "ddr4_vts25_v1" not in ADMITTED_PROFILES:
        raise SystemExit("admitted DDR4 profile registry is inconsistent")
    if not DEFERRED_PROFILES:
        raise SystemExit("deferred profile set is empty")

    env = RowHammerTaskEnv(profile_id="hbm2_read_disturbance_v1")
    obs = env.reset(seed=9, episode_id="phase9_deferred_profile")
    if not obs.error or obs.error["code"] != "UNAVAILABLE_CAPABILITY":
        raise SystemExit("non-admitted profile did not fail closed")

    print("phase9 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

