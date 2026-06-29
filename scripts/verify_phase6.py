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


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase5.py"], cwd=ROOT, check=True)

    direct = RowHammerTaskEnv()
    direct.reset(seed=6, episode_id="phase6_direct")
    target = direct.disturbance.target_addr  # type: ignore[union-attr]
    threshold = direct.disturbance.known_threshold  # type: ignore[union-attr]
    left = target - direct.disturbance.row_bytes  # type: ignore[union-attr]
    right = target + direct.disturbance.row_bytes  # type: ignore[union-attr]
    for _ in range(threshold // 2):
        direct.step(
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
    direct_value = read_byte(direct, target)
    direct.close()

    script = RowHammerTaskEnv()
    script.reset(seed=6, episode_id="phase6_script")
    code = (
        "from rh_sdk import rh\n"
        f"for _ in range({threshold // 2}):\n"
        f"    rh.issue(commands=[{{'op':'RD','addr':{left}}}, {{'op':'RD','addr':{right}}}])\n"
    )
    obs = script.step(Phase2Action(tool="script.run", args={"language": "python-rh-sdk", "code": code}))
    script_value = read_byte(script, target)
    if obs.error or direct_value != script_value or script_value != 1:
        raise SystemExit("script path was not trace-equivalent to direct tools")

    bad = script.step(Phase2Action(tool="script.run", args={"language": "python-rh-sdk", "code": "import os\n"}))
    if not bad.error or bad.error["code"] != "SANDBOX_VIOLATION":
        raise SystemExit("sandbox did not reject host-access import")
    script.close()

    print("phase6 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

