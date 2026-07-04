#!/usr/bin/env python3
"""Phase 18 admission gate: OS-isolated script sandbox and IPC broker."""

from __future__ import annotations

import base64
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerTaskEnv  # noqa: E402
from rowhammer_env.script_sandbox import sandbox_attestation  # noqa: E402


def _require_worker() -> None:
    worker = ROOT / "build/phase2/ramulator_worker"
    config = ROOT / "build/phase2/p2_external_ddr4.yaml"
    if not worker.is_file() or not config.is_file():
        raise SystemExit("Phase 2 worker/config is not built; run scripts/build_phase2.py first")


def check_attestation() -> None:
    attestation = sandbox_attestation()
    required = ("host_fs_blocked", "proc_pagemap_blocked", "dev_mem_blocked", "dev_kvm_blocked", "network_blocked")
    if (
        attestation.get("runtime") != "unshare+bwrap"
        or attestation.get("control_usr_readable") is not True
        or not all(attestation.get(key) is True for key in required)
    ):
        raise SystemExit(f"sandbox attestation failed: {attestation}")
    print("  attestation: unshare+bwrap blocks host fs, /proc/self/pagemap, devices, and network")


def check_escape_attempts() -> None:
    cases = {
        "import os\n": "SANDBOX_VIOLATION",
        "open('/etc/hostname').read()\n": "SANDBOX_VIOLATION",
        "print.__self__.__import__('os')\n": "SANDBOX_VIOLATION",
        "while True:\n    pass\n": "SCRIPT_TIMEOUT",
    }
    for code, expected in cases.items():
        env = RowHammerTaskEnv()
        obs = env.step(Phase2Action(tool="script.run", args={"code": code, "timeout_ms": 250}))
        if not obs.error or obs.error["code"] != expected:
            raise SystemExit(f"escape case returned {obs.error}, expected {expected}: {code!r}")
        env.close()
    print("  escapes: imports, filesystem/builtin escapes, and infinite loops fail closed")


def check_trace_and_reward_equivalence() -> None:
    direct = RowHammerTaskEnv()
    obs = direct.reset(seed=18, episode_id="phase18_direct")
    if obs.error:
        raise SystemExit(f"direct reset failed: {obs.error}")
    dist = direct.disturbance
    target = dist.target_addr
    threshold = dist.known_threshold
    left = target - dist.row_bytes
    right = target + dist.row_bytes
    direct_last = None
    for _ in range(threshold // 2):
        direct_last = direct.step(
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
        if direct_last.error:
            raise SystemExit(f"direct issue failed: {direct_last.error}")
    direct_value = _read_byte(direct, target)
    direct.close()

    script = RowHammerTaskEnv()
    obs = script.reset(seed=18, episode_id="phase18_script")
    if obs.error:
        raise SystemExit(f"script reset failed: {obs.error}")
    code = (
        "from rh_sdk import rh\n"
        f"for _ in range({threshold // 2}):\n"
        f"    rh.issue(commands=[{{'op':'RD','addr':{left}}}, {{'op':'RD','addr':{right}}}])\n"
    )
    script_obs = script.step(Phase2Action(tool="script.run", args={"language": "python-rh-sdk", "code": code, "timeout_ms": 10000}))
    if script_obs.error:
        raise SystemExit(f"script.run failed: {script_obs.error}")
    script_value = _read_byte(script, target)
    script_trace = script_obs.feedback["script"]["observations"][-1]["feedback"]["trace_tail"]
    if direct_value != script_value or script_value == 0:
        raise SystemExit(f"script/direct final values differ: direct={direct_value} script={script_value}")
    if script_obs.reward != 1.0 or not script_obs.done:
        raise SystemExit(f"script did not reach trusted reward: reward={script_obs.reward} done={script_obs.done}")
    if script_trace != direct_last.feedback["trace_tail"]:
        raise SystemExit("script trace tail differs from direct tool sequence")
    script.close()
    print("  equivalence: script IPC calls match direct trace tail and trusted reward")


def _read_byte(env: RowHammerTaskEnv, addr: int) -> int:
    obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": 1}))
    if obs.error:
        raise SystemExit(f"read failed: {obs.error}")
    return base64.b64decode(obs.data_b64 or "")[0]


def main() -> int:
    _require_worker()
    subprocess.run([sys.executable, "-B", "-m", "unittest", "tests.test_phase18"], cwd=ROOT, check=True)
    print("phase18 checks:")
    check_attestation()
    check_escape_attempts()
    check_trace_and_reward_equivalence()
    print("phase18 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
