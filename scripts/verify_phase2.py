#!/usr/bin/env python3
from __future__ import annotations

import base64
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerEnv  # noqa: E402


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def read(env: RowHammerEnv, addr: int, length: int):
    return env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": length}))


def write(env: RowHammerEnv, addr: int, data: bytes):
    return env.step(Phase2Action(tool="dram.write", args={"addr": {"kind": "logical", "addr": addr}, "data_b64": b64(data)}))


def run_sequence(episode_id: str):
    env = RowHammerEnv()
    env.reset(episode_id=episode_id)
    payload = b"rh2!"
    w = write(env, 4096, payload)
    r = read(env, 4096, len(payload))
    wait = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "WAIT", "cycles": 7}]}))
    rd = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "RD", "addr": {"kind": "logical", "addr": 4096}}]}))
    out = (
        w.last_action.get("cycle_delta"),
        r.data_b64,
        wait.last_action.get("cycle_delta"),
        rd.feedback["trace_tail"][-1]["op"],
        rd.cycle,
    )
    env.close()
    return out


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase1.py"], cwd=ROOT, check=True)
    if not (ROOT / "build/phase2/ramulator_worker").is_file():
        raise SystemExit("missing Phase 2 worker; run scripts/build_phase2.py")

    env = RowHammerEnv()
    obs = env.reset(episode_id="phase2_verify")
    if obs.error:
        raise SystemExit(obs.error)
    payload = b"rh2!"
    if write(env, 4096, payload).error:
        raise SystemExit("write failed")
    got = read(env, 4096, len(payload))
    if base64.b64decode(got.data_b64 or "") != payload:
        raise SystemExit("read did not return written bytes")
    wait = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "WAIT", "cycles": 7}]}))
    if wait.last_action.get("cycle_delta") != 7:
        raise SystemExit("WAIT did not advance by requested cycles")
    rd = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "RD", "addr": {"kind": "logical", "addr": 4096}}]}))
    if rd.feedback["trace_tail"][-1]["op"] != "RD":
        raise SystemExit("RD command did not enter public trace")
    bad = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "ACT", "addr": {"kind": "logical", "addr": 4096}}]}))
    if not bad.error or bad.error["code"] != "ILLEGAL_COMMAND":
        raise SystemExit("unsupported direct command did not fail closed")
    env.close()

    env2 = RowHammerEnv()
    env2.reset(episode_id="phase2_reset_check")
    empty = read(env2, 4096, len(payload))
    if base64.b64decode(empty.data_b64 or "") == payload:
        raise SystemExit("fresh episode leaked previous sparse memory overlay")
    env2.close()

    if run_sequence("phase2_det_a") != run_sequence("phase2_det_b"):
        raise SystemExit("same Phase 2 command sequence was not deterministic")

    print("phase2 verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
