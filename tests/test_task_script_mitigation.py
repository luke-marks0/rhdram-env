from __future__ import annotations

import unittest

from rowhammer_env import Phase2Action, RowHammerTaskEnv


class TaskScriptMitigationTests(unittest.TestCase):
    def test_finish_before_success_rewards_zero(self) -> None:
        env = RowHammerTaskEnv()
        env.reset(seed=1)
        obs = env.step(Phase2Action(tool="episode.finish", args={}))
        self.assertTrue(obs.done)
        self.assertEqual(obs.reward, 0.0)

    def test_script_rejects_import(self) -> None:
        env = RowHammerTaskEnv()
        env.reset(seed=1)
        obs = env.step(Phase2Action(tool="script.run", args={"code": "import os\n"}))
        self.assertEqual(obs.error["code"], "SANDBOX_VIOLATION")
        env.close()

    def test_unavailable_mitigation_fails_closed(self) -> None:
        env = RowHammerTaskEnv(mitigation={"name": "para", "params": {}})
        obs = env.reset(seed=1)
        self.assertEqual(obs.error["code"], "UNAVAILABLE_CAPABILITY")


if __name__ == "__main__":
    unittest.main()

