from __future__ import annotations

import base64
import pathlib
import unittest

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv


ROOT = pathlib.Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "build/phase2/ramulator_worker").is_file(), "Phase 2 worker not built")
class DisturbanceTests(unittest.TestCase):
    def test_write_restores_flip_overlay(self) -> None:
        env = RowHammerDisturbanceEnv()
        env.reset(seed=1)
        target = env.disturbance.target_addr
        env.disturbance.flips[target] = 0
        obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": target}, "length": 1}))
        self.assertEqual(base64.b64decode(obs.data_b64 or ""), b"\x01")
        env.step(
            Phase2Action(
                tool="dram.write",
                args={"addr": {"kind": "logical", "addr": target}, "data_b64": base64.b64encode(b"\x00").decode()},
            )
        )
        obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": target}, "length": 1}))
        self.assertEqual(base64.b64decode(obs.data_b64 or ""), b"\x00")
        env.close()


if __name__ == "__main__":
    unittest.main()

