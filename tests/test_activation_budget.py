"""Activation-budget enforcement inside ``dram.issue`` (P26 follow-up).

A single ``dram.issue`` may not spend more activations than the episode's remaining
ACT budget. ``_issue`` drives the worker one primitive at a time, so it stops issuing
once the budget is spent — a monolithic ``HAMMER`` is truncated at the budget instead
of running to completion and crediting an over-budget flip. Without this, a
timing-blind policy could brute-force a flip by hammering *every* candidate in one
call regardless of budget, and the bank-conflict timing channel would carry no
practical advantage. With it, a discovery budget is a real constraint the timing
channel is needed to respect (SPEC §8: each action's ACT count is budgeted).
"""

from __future__ import annotations

import pathlib
import unittest

from rowhammer_env import Phase2Action, RowHammerTaskEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class ActivationBudgetTests(unittest.TestCase):
    def _issue(self, env, commands):
        return env.step(Phase2Action(tool="dram.issue", args={"commands": commands}))

    def test_oversized_hammer_is_truncated_without_crediting_a_flip(self) -> None:
        # Hammering ALL hidden_adjacency candidates needs tens of thousands of ACTs;
        # under a tight ACT budget the issue truncates exactly at the budget and no
        # over-budget flip is credited.
        env = RowHammerTaskEnv(task={"family": "hidden_adjacency", "difficulty": "medium", "id": "ab"},
                               budgets={"tool_calls": 10, "acts": 3000, "cycles": 240_000_000})
        try:
            obs = env.reset(seed=1)
            candidates = obs.metadata["candidates"]
            step = self._issue(env, [{"op": "HAMMER", "rows": candidates, "pairs": 5000}])
            self.assertEqual(step.reward, 0.0)
            self.assertEqual((step.error or {}).get("code"), "BUDGET_EXCEEDED")
            self.assertTrue(step.done)
            # Spent at most the budget (the per-primitive loop overshoots by <=1 ACT),
            # nowhere near the >20k ACTs the full hammer would have consumed.
            self.assertLessEqual(int(step.public_counters["acts"]), 3001)
        finally:
            env.close()

    def test_same_aggressors_flip_when_the_budget_is_ample(self) -> None:
        # The truncation above fails for lack of budget, not because the addresses are
        # wrong: the true aggressors flip the victim once the budget can pay for them.
        env = RowHammerTaskEnv(task={"family": "hidden_adjacency", "difficulty": "medium", "id": "ab"},
                               budgets={"tool_calls": 10, "acts": 60_000, "cycles": 240_000_000})
        try:
            obs = env.reset(seed=1)
            ct = env._compiled
            meta = obs.metadata["candidates"]
            aggressors = [meta[i] for i, c in enumerate(ct.candidates) if c.is_aggressor]
            step = self._issue(env, [{"op": "HAMMER", "rows": aggressors, "pairs": env.disturbance.known_threshold}])
            self.assertEqual(step.reward, 1.0)
            self.assertIsNone(step.error)
        finally:
            env.close()

    def test_family_without_acts_budget_is_not_truncated(self) -> None:
        # The known-target (legacy-budget) families do not budget activations, so the
        # activation ceiling is absent and their hammers run freely — the enforcement
        # is scoped to families that actually cap ACTs (regression guard).
        env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
        try:
            env.reset(seed=1)
            self.assertNotIn("acts", env.budget_remaining)
            self.assertIsNone(env._issue_acts_ceiling())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
