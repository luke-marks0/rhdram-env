from __future__ import annotations

import base64
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
CONFIG = ROOT / "build/phase2/p2_external_ddr4.yaml"

from rowhammer_env import Phase2Action, RowHammerEnv


class Phase2EnvTests(unittest.TestCase):
    def test_missing_worker_fails_closed_on_reset(self) -> None:
        env = RowHammerEnv(worker_path=ROOT / "missing-worker", config_path=ROOT / "missing-config.yaml")
        obs = env.reset()
        self.assertEqual(obs.error["code"], "UNAVAILABLE_CAPABILITY")

    @unittest.skipUnless(WORKER.is_file() and CONFIG.is_file(), "Phase 2 worker/config not built")
    def test_read_write_overlay(self) -> None:
        env = RowHammerEnv()
        env.reset()
        data = base64.b64encode(b"abcd").decode()
        self.assertIsNone(env.step(Phase2Action(tool="dram.write", args={"addr": {"kind": "logical", "addr": 8192}, "data_b64": data})).error)
        obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": 8192}, "length": 4}))
        self.assertEqual(base64.b64decode(obs.data_b64 or ""), b"abcd")
        env.close()


if __name__ == "__main__":
    unittest.main()
