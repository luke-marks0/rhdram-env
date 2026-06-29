#!/usr/bin/env python3
from __future__ import annotations

import base64
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv


def rd(addr: int) -> dict:
    return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}


def read_byte(env: RowHammerDisturbanceEnv, addr: int) -> int:
    obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": 1}))
    if obs.error:
        raise SystemExit(obs.error)
    return base64.b64decode(obs.data_b64 or "")[0]


def hammer(env: RowHammerDisturbanceEnv, left: int, right: int, pairs: int) -> None:
    for _ in range(pairs):
        obs = env.step(Phase2Action(tool="dram.issue", args={"commands": [rd(left), rd(right)]}))
        if obs.error:
            raise SystemExit(obs.error)


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase2.py"], check=True)

    env = RowHammerDisturbanceEnv()
    obs = env.reset(seed=4, episode_id="phase4_flip")
    if obs.error:
        raise SystemExit(obs.error)
    target = env.disturbance.target_addr  # type: ignore[union-attr]
    threshold = env.disturbance.known_threshold  # type: ignore[union-attr]
    left = target - env.disturbance.row_bytes  # type: ignore[union-attr]
    right = target + env.disturbance.row_bytes  # type: ignore[union-attr]

    if read_byte(env, target) != 0:
        raise SystemExit("fresh target byte was not zero")
    hammer(env, left, right, threshold // 2)
    if read_byte(env, target) != 1:
        raise SystemExit("known vulnerable row did not flip")
    env.step(
        Phase2Action(
            tool="dram.write",
            args={"addr": {"kind": "logical", "addr": target}, "data_b64": base64.b64encode(b"\x00").decode()},
        )
    )
    if read_byte(env, target) != 0:
        raise SystemExit("write did not restore flipped byte")
    env.close()

    control = RowHammerDisturbanceEnv()
    control.reset(seed=4, episode_id="phase4_control")
    hammer(control, left, right, threshold // 2 - 1)
    if read_byte(control, target) != 0:
        raise SystemExit("below-threshold control flipped")
    control.close()

    one_sided = RowHammerDisturbanceEnv()
    one_sided.reset(seed=4, episode_id="phase4_one_sided")
    for _ in range(threshold):
        obs = one_sided.step(Phase2Action(tool="dram.issue", args={"commands": [rd(left)]}))
        if obs.error:
            raise SystemExit(obs.error)
    if read_byte(one_sided, target) != 0:
        raise SystemExit("single-sided control flipped in double-sided fixture")
    one_sided.close()

    print(f"phase4 verification passed (threshold={threshold})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
