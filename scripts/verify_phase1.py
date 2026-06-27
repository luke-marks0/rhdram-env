#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def require(path: pathlib.Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing required Phase 1 artifact: {path.relative_to(ROOT)}")


def require_commit(path: pathlib.Path, expected: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise SystemExit(f"{path.relative_to(ROOT)} is {actual}, expected {expected}")


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase0.py"], cwd=ROOT, check=True)
    require(ROOT / "third_party/ramulator2")
    require(ROOT / "third_party/openenv/src")
    require(ROOT / "build/phase1/ramulator_external_smoke")
    require(ROOT / "build/phase1/p1_external_ddr4.yaml")
    require_commit(ROOT / "third_party/ramulator2", "278f1effc3838099a6ffe0ad5f9f572fea80c948")
    require_commit(ROOT / "third_party/openenv", "7449c5dfe375c4c6e6f0827826925a46efd9249f")

    sys.path.insert(0, str(ROOT / "third_party/openenv/src"))
    sys.path.insert(0, str(ROOT))
    from rowhammer_env import Phase1Action, RowHammerBootstrapEnv

    env = RowHammerBootstrapEnv()
    env.reset(episode_id="phase1_verify")
    obs = env.step(Phase1Action())
    if obs.error:
        raise SystemExit(f"phase1 step failed: {obs.error}")
    if not env.state.ramulator_bootstrap_complete:
        raise SystemExit("phase1 step did not mark bootstrap complete")
    if not obs.ramulator.get("write", {}).get("completed") or not obs.ramulator.get("read", {}).get("completed"):
        raise SystemExit("Ramulator read/write requests did not complete")

    print("phase1 verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
