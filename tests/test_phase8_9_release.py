from __future__ import annotations

import json
import unittest

from rowhammer_env import RowHammerTaskEnv
from rowhammer_env.profiles import load_profile


class AdvancedTaskTests(unittest.TestCase):
    def test_hidden_target_metadata_does_not_disclose_coordinates(self) -> None:
        env = RowHammerTaskEnv(task={"family": "hidden_target"})
        obs = env.reset(seed=1)
        text = json.dumps(obs.metadata, sort_keys=True)
        self.assertNotIn("known_target_row", text)
        self.assertNotIn(str(env.disturbance.target_addr), text)  # type: ignore[union-attr]
        env.close()

    def test_any_flip_task_uses_trusted_flip_state(self) -> None:
        env = RowHammerTaskEnv(task={"family": "any_flip"})
        env.reset(seed=1)
        self.assertFalse(env._trusted_success())
        env.disturbance.flips[env.disturbance.target_addr] = 0  # type: ignore[union-attr]
        self.assertTrue(env._trusted_success())
        env.close()


class ProfileRegistryTests(unittest.TestCase):
    def test_admitted_profile_loads(self) -> None:
        self.assertEqual(load_profile("ddr4_vts25_v1")["standard"], "DDR4")

    def test_deferred_profile_fails_closed(self) -> None:
        env = RowHammerTaskEnv(profile_id="hbm2_read_disturbance_v1")
        obs = env.reset(seed=1)
        self.assertEqual(obs.error["code"], "UNAVAILABLE_CAPABILITY")


if __name__ == "__main__":
    unittest.main()

