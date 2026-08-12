"""Budget enforcement at the env boundary (P26 follow-up).

A single ``dram.issue`` may not spend more activations than the episode's remaining
ACT budget. ``_issue`` drives the worker one primitive at a time and checks the
ceiling *before* each primitive is sent, so a monolithic ``HAMMER`` is truncated at
the budget instead of running to completion and crediting an over-budget flip.
Without this, a timing-blind policy could brute-force a flip by hammering *every*
candidate in one call regardless of budget, and the bank-conflict timing channel
would carry no practical advantage. With it, a discovery budget is a real constraint
the timing channel is needed to respect (SPEC §8: each action's ACT count is
budgeted).

The same honesty has to hold at the episode boundary: once the episode is over, or
once no tool call is left to pay for one, the env refuses the action *before*
dispatching it, so budget honesty is a property of the env and not of whichever
driver happens to stop on ``done``.

# @spec:invariant-budget-honesty @spec:rl-episode-termination
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
            # Spent at most the budget — the ceiling is checked before each primitive
            # is sent, so it is never crossed — and nowhere near the >20k ACTs the
            # full hammer would have consumed.
            self.assertLessEqual(int(step.public_counters["acts"]), 3000)
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

    def test_issue_at_zero_remaining_acts_sends_nothing(self) -> None:
        # The budget-honesty boundary: with the ACT budget at exactly 0 the ceiling
        # equals the counter, so the *first* primitive is already over budget. It
        # must never reach the worker — a post-issue check could only notice the
        # activation after the disturbance model had committed it.
        # dram.read is not ceiling-guarded (it is charged post-hoc), so reading two
        # distinct rows is how an episode legitimately lands on a live 0.
        env = RowHammerTaskEnv(task={"family": "hidden_adjacency", "difficulty": "medium", "id": "ab"},
                               budgets={"tool_calls": 8, "acts": 2, "cycles": 240_000_000})
        try:
            env.reset(seed=1)
            row_bytes = env._compiled.row_bytes
            base = env._compiled.target_addr
            for i in (1, 2):
                read = env.step(Phase2Action(
                    tool="dram.read",
                    args={"addr": {"kind": "logical", "addr": base + i * row_bytes}, "length": 1},
                ))
            self.assertEqual(read.metadata["budget_remaining"]["acts"], 0)
            self.assertFalse(read.done)
            spent = int(read.public_counters["acts"])
            cycle = read.cycle

            step = self._issue(env, [{"op": "HAMMER", "rows": [base + 3 * row_bytes, base + 4 * row_bytes],
                                      "pairs": 100}])
            self.assertEqual((step.error or {}).get("code"), "BUDGET_EXCEEDED")
            self.assertTrue(step.done)
            # Nothing was issued: neither the activation counter nor the simulator
            # clock moved, and the budget was not driven negative.
            self.assertEqual(int(step.public_counters["acts"]), spent)
            self.assertEqual(step.cycle, cycle)
            self.assertEqual(step.metadata["budget_remaining"]["acts"], 0)
        finally:
            env.close()

    def test_step_after_the_episode_is_over_is_refused(self) -> None:
        # Budget honesty must not depend on the driver: once the episode is done the
        # env itself refuses to act, so a caller that ignores ``done`` (the script
        # broker forwards it without acting on it) cannot keep hammering.
        env = RowHammerTaskEnv(task={"family": "hidden_adjacency", "difficulty": "medium", "id": "ab"},
                               budgets={"tool_calls": 8, "acts": 2, "cycles": 240_000_000})
        try:
            env.reset(seed=1)
            row_bytes = env._compiled.row_bytes
            base = env._compiled.target_addr
            done = self._issue(env, [{"op": "HAMMER", "rows": [base - row_bytes, base + row_bytes], "pairs": 100}])
            self.assertTrue(done.done)
            budget_at_end = dict(done.metadata["budget_remaining"])

            after = self._issue(env, [{"op": "HAMMER", "rows": [base - row_bytes, base + row_bytes], "pairs": 100}])
            self.assertEqual((after.error or {}).get("code"), "UNAVAILABLE_CAPABILITY")
            self.assertTrue(after.done)
            self.assertEqual(after.reward, 0.0)
            # Refused before dispatch: no budget was charged for the refusal itself.
            self.assertEqual(after.metadata["budget_remaining"], budget_at_end)
            # ... and episode.finish cannot resurrect a reward either.
            finish = env.step(Phase2Action(tool="episode.finish", args={}))
            self.assertEqual((finish.error or {}).get("code"), "UNAVAILABLE_CAPABILITY")
            self.assertEqual(finish.reward, 0.0)
        finally:
            env.close()

    def test_step_without_a_tool_call_left_is_refused_before_dispatch(self) -> None:
        # Same defect shape on the tool_calls axis: charging after execution lets the
        # call that drives the budget to -1 run (and its flips count). The last legal
        # call is the one that lands on 0; the next is refused, not executed.
        env = RowHammerTaskEnv(task={"family": "hidden_adjacency", "difficulty": "medium", "id": "ab"},
                               budgets={"tool_calls": 2, "acts": 60_000, "cycles": 240_000_000})
        try:
            env.reset(seed=1)
            base = env._compiled.target_addr
            for _ in range(2):
                obs = env.step(Phase2Action(
                    tool="dram.read", args={"addr": {"kind": "logical", "addr": base}, "length": 1}
                ))
                self.assertIsNone(obs.error)
            self.assertEqual(obs.metadata["budget_remaining"]["tool_calls"], 0)
            acts = int(obs.public_counters["acts"])

            refused = env.step(Phase2Action(
                tool="dram.read", args={"addr": {"kind": "logical", "addr": base}, "length": 1}
            ))
            self.assertEqual((refused.error or {}).get("code"), "BUDGET_EXCEEDED")
            self.assertTrue(refused.done)
            self.assertEqual(refused.metadata["budget_remaining"]["tool_calls"], 0)
            self.assertEqual(env._acts_issued(), acts)
        finally:
            env.close()

    def test_a_rejected_action_does_not_end_the_episode(self) -> None:
        # A malformed command rejects the *action*: the episode keeps running and the
        # observation says so, so a driver that stops on ``done`` (every rollout loop
        # in ``llm/``) does not abandon an episode over a typo.
        env = RowHammerTaskEnv(task={"family": "hidden_adjacency", "difficulty": "medium", "id": "ab"},
                               budgets={"tool_calls": 8, "acts": 60_000, "cycles": 240_000_000})
        try:
            env.reset(seed=1)
            bad = self._issue(env, [{"op": "ACT", "addr": {"kind": "logical", "addr": 0}}])
            self.assertEqual((bad.error or {}).get("code"), "ILLEGAL_COMMAND")
            self.assertFalse(bad.done)
            good = env.step(Phase2Action(
                tool="dram.read",
                args={"addr": {"kind": "logical", "addr": env._compiled.target_addr}, "length": 1},
            ))
            self.assertIsNone(good.error)
        finally:
            env.close()

    def test_a_failed_reset_ends_the_episode(self) -> None:
        # The other half of the same contract: a code that only *rejects* an action
        # mid-episode is still terminal when it comes from ``reset``, because an
        # episode that cannot start at all is ended by ``reset`` itself.
        env = RowHammerTaskEnv(mitigation={"name": "para", "params": {}})
        try:
            obs = env.reset(seed=1)
            self.assertEqual((obs.error or {}).get("code"), "UNAVAILABLE_CAPABILITY")
            self.assertTrue(obs.done)
            # ... and the env holds itself to it: the next step is refused.
            after = env.step(Phase2Action(tool="dram.info", args={}))
            self.assertEqual((after.error or {}).get("code"), "UNAVAILABLE_CAPABILITY")
            self.assertTrue(after.done)
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
