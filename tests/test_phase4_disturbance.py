from __future__ import annotations

import base64
import pathlib
import unittest

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv


ROOT = pathlib.Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "build/phase2/ramulator_worker").is_file(), "Phase 2 worker not built")
class DisturbanceTests(unittest.TestCase):
    # @spec:tool-dram-write @spec:sim-latent-vulnerability
    def test_write_records_worker_decoded_pattern_for_both_write_routes(self) -> None:
        for tool, args_for in (
            (
                "dram.write",
                lambda addr, data: {
                    "addr": {"kind": "logical", "addr": addr},
                    "data_b64": data,
                },
            ),
            (
                "dram.issue",
                lambda addr, data: {
                    "commands": [
                        {
                            "op": "WR",
                            "addr": {"kind": "logical", "addr": addr},
                            "data_b64": data,
                        }
                    ]
                },
            ),
        ):
            with self.subTest(tool=tool):
                env = RowHammerDisturbanceEnv()
                reset = env.reset(seed=14)
                self.assertIsNone(reset.error)
                assert env.disturbance is not None
                row = 5000
                victim_addr = row * env.disturbance.row_bytes
                row_key = (0, 0, 0, 0, row)
                data_b64 = base64.b64encode(b"\xff" * 64).decode()
                env.disturbance.flips[victim_addr] = 0

                write = env.step(Phase2Action(tool=tool, args=args_for(victim_addr, data_b64)))
                self.assertIsNone(write.error)
                self.assertEqual(env.disturbance._row_pattern[row_key], "all_ones")
                self.assertNotIn(victim_addr, env.disturbance.flips)

                # Activating the preceding physical row materializes this victim;
                # its latent state must now come from the all-ones stratum.
                probe = env.step(
                    Phase2Action(
                        tool="dram.read",
                        args={
                            "addr": {
                                "kind": "logical",
                                "addr": victim_addr - env.disturbance.row_bytes,
                            },
                            "length": 1,
                        },
                    )
                )
                self.assertIsNone(probe.error)
                victim = env.disturbance.victims[row_key]
                self.assertEqual(victim.data_pattern, "all_ones")
                self.assertEqual(victim.direction, "1->0")
                self.assertEqual(
                    victim.threshold,
                    env.disturbance._sample_threshold(row_key, "double", "all_ones"),
                )
                env.close()

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
