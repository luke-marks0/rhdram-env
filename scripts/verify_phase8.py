#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerTaskEnv


def rd(addr: int) -> dict:
    return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}


def hammer_to_flip(env: RowHammerTaskEnv):
    target = env.disturbance.target_addr  # type: ignore[union-attr]
    row_bytes = env.disturbance.row_bytes  # type: ignore[union-attr]
    threshold = env.disturbance.known_threshold  # type: ignore[union-attr]
    obs = None
    for _ in range(threshold // 2):
        obs = env.step(Phase2Action(tool="dram.issue", args={"commands": [rd(target - row_bytes), rd(target + row_bytes)]}))
        if obs.error:
            raise SystemExit(obs.error)
    return obs


def expect_reward(task: dict, seed: int = 8) -> None:
    env = RowHammerTaskEnv(task=task)
    obs = env.reset(seed=seed, episode_id=f"phase8_{task['family']}")
    if obs.error:
        raise SystemExit(obs.error)
    final = hammer_to_flip(env)
    if final.reward != 1.0 or not final.done:
        raise SystemExit(f"{task['family']} did not reward trusted success")
    env.close()


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase7.py"], cwd=ROOT, check=True)

    for family in ("any_flip", "target_cell", "pattern_target", "mitigation_aware"):
        task = {"family": family}
        if family == "target_cell":
            task["bit"] = 0
        expect_reward(task)

    env = RowHammerTaskEnv(task={"family": "hidden_target"})
    obs = env.reset(seed=8, episode_id="phase8_hidden")
    if obs.error:
        raise SystemExit(obs.error)
    text = json.dumps(obs.metadata, sort_keys=True)
    if "known_target_row" in text or str(env.disturbance.target_addr) in text:  # type: ignore[union-attr]
        raise SystemExit("hidden target metadata leaks target coordinates")
    env.close()

    print("phase8 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

