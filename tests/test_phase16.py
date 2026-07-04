from __future__ import annotations

import base64
import pathlib
import tempfile
import unittest

import yaml

from rowhammer_env import Phase2Action, RowHammerEnv, RowHammerTaskEnv
from rowhammer_env.disturbance import DisturbanceEngine
from rowhammer_env.geometry import Geometry
from rowhammer_env.mitigations import (
    ADMITTED,
    public_mitigation_capabilities,
    ramulator_controller_plugins,
    unavailable_mitigation_names,
    worker_config_for_mitigation,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
CONFIG = ROOT / "build/phase2/p2_external_ddr4.yaml"

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


class MitigationRegistryTests(unittest.TestCase):
    def test_capability_discovery_lists_only_admitted_mitigations(self) -> None:
        names = [item["name"] for item in public_mitigation_capabilities()]
        self.assertEqual(names, ["none", "oracle"])
        for unavailable in unavailable_mitigation_names():
            self.assertNotIn(unavailable, names)

    def test_known_unavailable_mitigations_fail_closed_before_worker_start(self) -> None:
        for name in ("para", "graphene", "twice", "blockhammer", "prac", "hydra", "rrs", "aqua", "rfm", "custom"):
            env = RowHammerTaskEnv(mitigation={"name": name, "params": {}})
            obs = env.reset(seed=16)
            self.assertEqual(obs.error["code"], "UNAVAILABLE_CAPABILITY", name)
            self.assertEqual(obs.error["message"], name)

    def test_disturbance_engine_uses_registry_not_inline_allowlist(self) -> None:
        geo = Geometry(DDR4_INFO)
        self.assertIn("oracle", ADMITTED)
        self.assertEqual(DisturbanceEngine(geometry=geo, mitigation="oracle").mitigation, "oracle")
        with self.assertRaises(ValueError) as ctx:
            DisturbanceEngine(geometry=geo, mitigation="para")
        self.assertEqual(str(ctx.exception), "UNAVAILABLE_CAPABILITY:para")

    def test_dram_info_exposes_only_admitted_capabilities(self) -> None:
        env = RowHammerEnv()
        obs = env.step(Phase2Action(tool="dram.info", args={}))
        names = {item["name"] for item in obs.metadata["mitigations"]}
        self.assertEqual(names, {"none", "oracle"})
        self.assertNotIn("para", names)

    def test_worker_config_surface_does_not_inject_missing_ramulator_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp) / "base.yaml"
            base.write_text(
                yaml.safe_dump(
                    {
                        "memory_system": {
                            "controllers": [
                                {"impl": "GenericDRAMController", "controller_plugins": [{"impl": "IssuedEventRecorder"}]}
                            ]
                        }
                    },
                    sort_keys=False,
                )
            )
            selected = worker_config_for_mitigation(base, {"name": "oracle", "params": {"tRH": 32}})
            self.assertEqual(selected, base)
            self.assertEqual(ramulator_controller_plugins({"name": "oracle", "params": {"tRH": 32}}), [])


@unittest.skipUnless(WORKER.is_file() and CONFIG.is_file(), "Phase 2 worker/config not built")
class MitigationWorkerTests(unittest.TestCase):
    def test_reset_and_dram_info_capabilities_match(self) -> None:
        env = RowHammerTaskEnv(mitigation={"name": "oracle", "params": {}})
        reset = env.reset(seed=16)
        self.assertIsNone(reset.error)
        info = env.step(Phase2Action(tool="dram.info", args={}))
        self.assertEqual(
            {m["name"] for m in reset.metadata["mitigations"]},
            {m["name"] for m in info.metadata["mitigations"]},
        )
        self.assertEqual(reset.metadata["mitigation"]["name"], "oracle")
        env.close()

    def test_mitigation_aware_oracle_protects_target(self) -> None:
        env = RowHammerTaskEnv(mitigation={"name": "oracle", "params": {}})
        obs = env.reset(seed=16)
        self.assertIsNone(obs.error)
        dist = env.disturbance
        assert dist is not None
        left = dist.target_addr - dist.row_bytes
        right = dist.target_addr + dist.row_bytes
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
            refreshes += int(obs.feedback.get("oracle_refreshes", 0))
            self.assertEqual(obs.reward, 0.0)
        read = env.step(
            Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": dist.target_addr}, "length": 1})
        )
        self.assertEqual(base64.b64decode(read.data_b64 or "")[0], 0)
        self.assertGreater(refreshes, 0)
        env.close()


if __name__ == "__main__":
    unittest.main()
