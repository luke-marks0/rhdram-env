#!/usr/bin/env python3
from __future__ import annotations

import base64
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerTaskEnv


def rd(addr: int) -> dict:
    return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}


def read_byte(env: RowHammerTaskEnv, addr: int) -> int:
    obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": 1}))
    if obs.error:
        raise SystemExit(obs.error)
    return base64.b64decode(obs.data_b64 or "")[0]


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase6.py"], cwd=ROOT, check=True)

    env = RowHammerTaskEnv(mitigation={"name": "oracle", "params": {}})
    obs = env.reset(seed=7, episode_id="phase7_oracle")
    if obs.error:
        raise SystemExit(obs.error)
    target = env.disturbance.target_addr  # type: ignore[union-attr]
    threshold = env.disturbance.known_threshold  # type: ignore[union-attr]
    left = target - env.disturbance.row_bytes  # type: ignore[union-attr]
    right = target + env.disturbance.row_bytes  # type: ignore[union-attr]
    oracle_refreshes = 0
    for _ in range(threshold):
        obs = env.step(Phase2Action(tool="dram.issue", args={"commands": [rd(left), rd(right)]}))
        oracle_refreshes += int(obs.feedback.get("oracle_refreshes", 0))
        if obs.reward:
            raise SystemExit("oracle mitigation allowed target reward")
    if read_byte(env, target) != 0 or oracle_refreshes == 0:
        raise SystemExit("oracle mitigation did not protect the target row")
    env.close()

    unavailable = RowHammerTaskEnv(mitigation={"name": "para", "params": {}})
    obs = unavailable.reset(seed=7)
    if not obs.error or obs.error["code"] != "UNAVAILABLE_CAPABILITY":
        raise SystemExit("unimplemented mitigation did not fail closed")

    print("phase7 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

