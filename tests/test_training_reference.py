"""Reference solver and negative controls on the real simulator (PoC scope controls).

These are the non-model half of the definition of done: the reference solver clears
every curriculum stage, the finish-only and below-threshold controls never fabricate a
flip, and the timing-blind (arithmetic-adjacency) control fails on the hidden-mapping
research task — so the timing channel is shown to be load-bearing where it matters.
"""
from __future__ import annotations

import pathlib
import unittest

from rowhammer_env.training import config as configlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class ReferenceAndControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from rowhammer_env.poc import PoCEnv, load_task

        cls.load_task = staticmethod(load_task)
        cls.stage_paths = dict(zip(configlib.STAGES, configlib.TASK_PATHS))
        cls.env = PoCEnv(task=load_task(configlib.TASK_PATHS[0]))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def _run(self, stage: str, policy, seed: int = 7):
        from rowhammer_env.training.rollout import run_episode

        task = self.load_task(self.stage_paths[stage])
        return run_episode(self.env, task=task, seed=seed, stage=stage, policy=policy, max_turns=40)

    def test_reference_clears_every_stage(self) -> None:
        from rowhammer_env.training.reference import ReferenceSolver

        for stage in configlib.STAGES:
            with self.subTest(stage=stage):
                self.assertEqual(self._run(stage, ReferenceSolver()).success, 1.0)

    def test_finish_only_never_succeeds(self) -> None:
        from rowhammer_env.training.reference import FinishOnlyControl

        for stage in configlib.STAGES:
            with self.subTest(stage=stage):
                self.assertEqual(self._run(stage, FinishOnlyControl()).success, 0.0)

    def test_below_threshold_never_succeeds(self) -> None:
        from rowhammer_env.training.reference import BelowThresholdControl

        for stage in configlib.STAGES:
            with self.subTest(stage=stage):
                self.assertEqual(self._run(stage, BelowThresholdControl()).success, 0.0)

    def test_timing_blind_fails_on_the_hidden_mapping_task(self) -> None:
        # On hidden_adjacency the arithmetic-adjacency guess lands in a different bank
        # under the secret mapper, so it cannot flip regardless of budget.
        from rowhammer_env.training.reference import ReferenceSolver, TimingBlindControl

        for stage in ("hidden_adjacency_easy", "hidden_adjacency_medium"):
            with self.subTest(stage=stage):
                self.assertEqual(self._run(stage, TimingBlindControl()).success, 0.0)
                self.assertEqual(self._run(stage, ReferenceSolver()).success, 1.0)


if __name__ == "__main__":
    unittest.main()
