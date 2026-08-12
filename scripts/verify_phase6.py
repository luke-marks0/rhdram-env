#!/usr/bin/env python3
from __future__ import annotations

import base64
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerTaskEnv


def read_byte(env: RowHammerTaskEnv, addr: int) -> int:
    obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": 1}))
    if obs.error:
        raise SystemExit(obs.error)
    return base64.b64decode(obs.data_b64 or "")[0]


def hammer_pair(env: RowHammerTaskEnv, left: int, right: int):
    return env.step(
        Phase2Action(
            tool="dram.issue",
            args={
                "commands": [
                    {"op": "RD", "addr": {"kind": "logical", "addr": left}},
                    {"op": "RD", "addr": {"kind": "logical", "addr": right}},
                ]
            },
        )
    )


def script_hammer(env: RowHammerTaskEnv, left: int, right: int, pairs: int):
    code = (
        "from rh_sdk import rh\n"
        f"for _ in range({pairs}):\n"
        f"    rh.issue(commands=[{{'op':'RD','addr':{left}}}, {{'op':'RD','addr':{right}}}])\n"
    )
    return env.step(Phase2Action(tool="script.run", args={"language": "python-rh-sdk", "code": code}))


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase5.py"], cwd=ROOT, check=True)

    direct = RowHammerTaskEnv()
    direct.reset(seed=6, episode_id="phase6_direct")
    target = direct.disturbance.target_addr  # type: ignore[union-attr]
    threshold = direct.disturbance.known_threshold  # type: ignore[union-attr]
    left = target - direct.disturbance.row_bytes  # type: ignore[union-attr]
    right = target + direct.disturbance.row_bytes  # type: ignore[union-attr]
    pairs = threshold // 2

    script = RowHammerTaskEnv()
    script.reset(seed=6, episode_id="phase6_script")

    # Stop one pair short of the flip on both paths and compare what a policy
    # actually observes while the episode is live: the victim byte read back
    # through dram.read, with no reward yet. The flipping pair ends the episode
    # (@spec:rl-episode-termination), so it is the terminal observation — not a
    # post-mortem read, which the env refuses — that the two paths are compared on.
    for _ in range(pairs - 1):
        obs = hammer_pair(direct, left, right)
        if obs.error:
            raise SystemExit(obs.error)
    early_script = script_hammer(script, left, right, pairs - 1)
    if early_script.error:
        raise SystemExit(early_script.error)
    direct_value = read_byte(direct, target)
    script_value = read_byte(script, target)
    if direct_value != script_value or direct_value != 0:
        raise SystemExit("script path did not leave the victim byte where the direct tools did")

    # The pair that crosses the threshold: same flip, same trusted reward, same
    # termination, whichever path issued it.
    direct_final = hammer_pair(direct, left, right)
    script_final = script_hammer(script, left, right, 1)
    if direct_final.reward != 1.0 or not direct_final.done:
        raise SystemExit("direct path did not reach the trusted reward on the target flip")
    if script_final.reward != direct_final.reward or not script_final.done:
        raise SystemExit("script path was not trace-equivalent to direct tools")
    direct.close()
    script.close()

    sandbox = RowHammerTaskEnv()
    sandbox.reset(seed=6, episode_id="phase6_sandbox")
    bad = sandbox.step(Phase2Action(tool="script.run", args={"language": "python-rh-sdk", "code": "import os\n"}))
    if not bad.error or bad.error["code"] != "SANDBOX_VIOLATION":
        raise SystemExit("sandbox did not reject host-access import")
    sandbox.close()

    print("phase6 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
