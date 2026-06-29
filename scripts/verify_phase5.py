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


def hammer(env: RowHammerTaskEnv, left: int, right: int, pairs: int):
    last = None
    for _ in range(pairs):
        last = env.step(Phase2Action(tool="dram.issue", args={"commands": [rd(left), rd(right)]}))
        if last.error:
            raise SystemExit(last.error)
    return last


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase4.py"], cwd=ROOT, check=True)
    env = RowHammerTaskEnv()
    obs = env.reset(seed=5, episode_id="phase5_known_target")
    if obs.error:
        raise SystemExit(obs.error)
    target = env.disturbance.target_addr  # type: ignore[union-attr]
    threshold = env.disturbance.known_threshold  # type: ignore[union-attr]
    left = target - env.disturbance.row_bytes  # type: ignore[union-attr]
    right = target + env.disturbance.row_bytes  # type: ignore[union-attr]

    early = hammer(env, left, right, threshold // 2 - 1)
    if early.reward != 0.0 or early.done:
        raise SystemExit("reward fired before target success")
    final = hammer(env, left, right, 1)
    if final.reward != 1.0 or not final.done:
        raise SystemExit("trusted reward did not fire on target flip")
    env.close()

    env = RowHammerTaskEnv()
    env.reset(seed=5, episode_id="phase5_finish")
    finished = env.step(Phase2Action(tool="episode.finish", args={}))
    if finished.reward != 0.0 or not finished.done:
        raise SystemExit("finish-before-success did not terminate with reward 0")
    env.close()

    env = RowHammerTaskEnv(budgets={"tool_calls": 1, "cycles": 5_000_000})
    env.reset(seed=5, episode_id="phase5_budget")
    env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": target}, "length": 1}))
    exhausted = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": target}, "length": 1}))
    if not exhausted.error or exhausted.error["code"] != "BUDGET_EXCEEDED":
        raise SystemExit("budget exhaustion did not fail closed")
    env.close()

    print("phase5 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

