from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]

from rowhammer_env import Phase1Action, RowHammerBootstrapEnv


class Phase1EnvTests(unittest.TestCase):
    def test_missing_worker_fails_closed(self) -> None:
        env = RowHammerBootstrapEnv(
            worker_path=ROOT / "missing-worker",
            config_path=ROOT / "missing-config.yaml",
        )
        env.reset()
        obs = env.step(Phase1Action())
        self.assertEqual(obs.error["code"], "UNAVAILABLE_CAPABILITY")
        self.assertTrue(obs.done)


if __name__ == "__main__":
    unittest.main()
