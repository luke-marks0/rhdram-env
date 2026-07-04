#!/usr/bin/env python3
"""Phase 16 admission gate: mitigation capability discovery and fail-closed use."""

from __future__ import annotations

import base64
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerTaskEnv  # noqa: E402
from rowhammer_env.disturbance import DisturbanceEngine  # noqa: E402
from rowhammer_env.geometry import Geometry  # noqa: E402
from rowhammer_env.mitigations import public_mitigation_capabilities  # noqa: E402


DDR4_INFO = {
    "standard": "DDR4",
    "tx_bytes": 64,
    "prefetch": 8,
    "channel_width": 64,
    "level_names": ["Channel", "Rank", "BankGroup", "Bank", "Row", "Column"],
    "level_sizes": [1, 1, 4, 4, 65536, 1024],
}


def act(row: int) -> dict:
    return {"op": "ACT", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": row, "row_hit": False}


def rd(addr: int) -> dict:
    return {"op": "RD", "addr": addr, "size": 64}


def check_registry() -> None:
    names = [item["name"] for item in public_mitigation_capabilities()]
    if names != ["none", "oracle"]:
        raise SystemExit(f"unexpected admitted mitigations: {names}")
    for name in ("para", "graphene", "twice", "blockhammer", "prac", "hydra", "rrs", "aqua", "rfm", "custom"):
        env = RowHammerTaskEnv(mitigation={"name": name, "params": {}})
        obs = env.reset(seed=16, episode_id=f"phase16_unavailable_{name}")
        if not obs.error or obs.error["code"] != "UNAVAILABLE_CAPABILITY":
            raise SystemExit(f"{name} did not fail closed: {obs.error}")
    print("  discovery: only none/oracle are advertised; known unvalidated mitigations fail closed")


def check_oracle_engine_conformance() -> None:
    eng = DisturbanceEngine(geometry=Geometry(DDR4_INFO), seed=16, mitigation="oracle")
    target = eng.known_target_row
    left = eng.target_addr - eng.row_bytes
    right = eng.target_addr + eng.row_bytes
    refreshes = 0
    for _ in range(eng.known_threshold):
        refreshes += eng.consume([act(target - 1)], rd(left)).oracle_refreshes
        refreshes += eng.consume([act(target + 1)], rd(right)).oracle_refreshes
    if refreshes <= 0 or eng.target_addr in eng.flips:
        raise SystemExit("oracle did not protect the target in the engine conformance trace")
    print(f"  oracle: reference-port protection fired {refreshes} times and prevented the target flip")


def check_oracle_worker_if_built() -> None:
    worker = ROOT / "build/phase2/ramulator_worker"
    config = ROOT / "build/phase2/p2_external_ddr4.yaml"
    if not worker.is_file() or not config.is_file():
        print("  worker: skipped (phase2 worker/config not built)")
        return
    env = RowHammerTaskEnv(mitigation={"name": "oracle", "params": {}})
    obs = env.reset(seed=16, episode_id="phase16_oracle_worker")
    if obs.error:
        raise SystemExit(f"oracle worker reset failed: {obs.error}")
    dist = env.disturbance
    target = dist.target_addr
    left = target - dist.row_bytes
    right = target + dist.row_bytes
    refreshes = 0
    for _ in range(dist.known_threshold):
        obs = env.step(
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
        if obs.error:
            raise SystemExit(f"oracle issue failed: {obs.error}")
        refreshes += int(obs.feedback.get("oracle_refreshes", 0))
        if obs.reward:
            raise SystemExit("oracle mitigation allowed target reward")
    obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": target}, "length": 1}))
    if obs.error:
        raise SystemExit(f"oracle read failed: {obs.error}")
    if base64.b64decode(obs.data_b64 or "")[0] != 0 or refreshes <= 0:
        raise SystemExit("oracle did not protect the target through the worker")
    info = env.step(Phase2Action(tool="dram.info", args={}))
    if {m["name"] for m in info.metadata.get("mitigations", [])} != {"none", "oracle"}:
        raise SystemExit(f"dram.info exposed wrong mitigations: {info.metadata.get('mitigations')}")
    env.close()
    print("  worker: oracle protects the task target and dram.info exposes only admitted mitigations")


def main() -> int:
    subprocess.run([sys.executable, "-B", "-m", "unittest", "tests.test_phase16"], cwd=ROOT, check=True)
    print("phase16 checks:")
    check_registry()
    check_oracle_engine_conformance()
    check_oracle_worker_if_built()
    print("phase16 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
