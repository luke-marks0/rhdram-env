"""Probe observability — full_trace wiring + per-issue timing digest (P22).

The DRAM bank-conflict timing side channel (TIER2_DISCOVERY_PLAN §0.2) only reaches
a policy if (a) the issued-event trace is disclosed (``full_trace``) and (b) a
multi-primitive ``dram.issue`` no longer drops every intermediate primitive's
timing (0.2.2). These tests drive the *real* ``RowHammerTaskEnv.step()`` and assert:

* a same-bank alternating probe forces new ACTs (nonzero ``acts_delta`` / large
  ``cycles_delta``) while a different-bank probe does not (~0) — the real
  discriminator, since a policy ``RD``'s own ``row_hit`` is always true (0.2.1);
* a 25k-activation ``HAMMER`` returns a bounded ``trace_tail`` and its
  ``timing_digest.acts_delta`` reconciles with the cumulative
  ``public_counters.acts``;
* the digest and trace carry no decoded coordinates and no hidden linear address,
  and the digest is absent at feedback levels that do not disclose the trace.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

from rowhammer_env import Phase2Action, RowHammerTaskEnv
from rowhammer_env.phase2_env import ISSUE_TRACE_TAIL_CAP
from rowhammer_env.tasks.compiler import TaskSpec
from rowhammer_env.tools.addressing import COORD_KEYS

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
UNKNOWN_ADJ = ROOT / "configs/tasks/unknown_adjacency.yaml"

# A logical_only + full_trace discovery task with generous budgets, so a big probe
# or HAMMER runs to completion without a BUDGET_EXCEEDED short-circuit.
DISCOVERY_TASK = {
    "id": "ddr4_probe_signal_v1",
    "family": "unknown_adjacency",
    "profile": "ddr4_vts25_v1",
    "standard": "DDR4",
    "mitigation": {"name": "none", "params": {}},
    "disclosure": {
        "mapping": "logical_only",
        "adjacency": "candidate_set",
        "victim": "row_handle",
        "profile": "public_profile_id",
        "feedback": "full_trace",
    },
    "objective": {"type": "target_row_flip", "target": "row_handle"},
    "budgets": {"tool_calls": 500, "acts": 400_000, "cycles": 60_000_000, "script_ms": 0},
    "reward": "sparse_success",
}


class ConfigWiringTests(unittest.TestCase):
    def test_unknown_adjacency_config_enables_full_trace(self) -> None:
        # P22 task 1: the shipped discovery config discloses the trace.
        doc = yaml.safe_load(UNKNOWN_ADJ.read_text())
        self.assertEqual(TaskSpec.from_config(doc).disclosure.feedback, "full_trace")


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class ProbeSignalTests(unittest.TestCase):
    def _issue(self, env, commands):
        return env.step(Phase2Action(tool="dram.issue", args={"commands": commands}))

    def _rd(self, addr: int) -> dict:
        return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}

    def _addrs(self, env):
        """Two addresses same-bank-adjacent to A, and one in a different bank."""
        m = env.address_mapper
        base = {"channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": 200, "column": 0}
        a = m.encode(base)
        same_bank = m.encode({**base, "row": 201})  # +row_stride: bank bits unchanged
        diff_bank = m.encode({**base, "bank": 1})   # flips a bank bit -> different flat_bank
        return a, same_bank, diff_bank

    def test_same_bank_probe_forces_acts_different_bank_does_not(self) -> None:
        env = RowHammerTaskEnv(task=DISCOVERY_TASK)
        try:
            env.reset(seed=7)
            a, same_bank, diff_bank = self._addrs(env)

            def probe(b: int) -> dict:
                # Warm both rows open (0.2), then alternate: same-bank evicts on
                # every access (new ACTs), different-bank stays open (row hits).
                self._issue(env, [self._rd(a), self._rd(b)])
                obs = self._issue(env, [{"op": "HAMMER", "rows": [a, b], "pairs": 8}])
                return obs.feedback["timing_digest"]

            same = probe(same_bank)
            diff = probe(diff_bank)

            self.assertGreater(same["acts_delta"], 0)
            self.assertEqual(diff["acts_delta"], 0)
            self.assertGreater(same["cycles_delta"], diff["cycles_delta"])
        finally:
            env.close()

    def test_probe_trace_has_no_coordinates(self) -> None:
        env = RowHammerTaskEnv(task=DISCOVERY_TASK)
        try:
            env.reset(seed=7)
            a, same_bank, _ = self._addrs(env)
            obs = self._issue(env, [self._rd(a), self._rd(same_bank), self._rd(a)])
            trace = obs.feedback["trace_tail"]
            self.assertTrue(trace, "full_trace should surface the probe's events")
            for event in trace:
                for coord in COORD_KEYS:
                    self.assertNotIn(coord, event)
            # The digest is counts/clocks only — no coordinate ever appears in it.
            digest = obs.feedback["timing_digest"]
            self.assertEqual(
                set(digest), {"acts_delta", "cycles_delta", "first_clk", "last_clk", "per_addr_hits"}
            )
        finally:
            env.close()

    def test_hammer_trace_bounded_and_acts_reconcile(self) -> None:
        env = RowHammerTaskEnv(task=DISCOVERY_TASK)
        try:
            env.reset(seed=7)
            a, same_bank, _ = self._addrs(env)
            warm = self._issue(env, [self._rd(a)])
            acts_before = int(warm.public_counters["acts"])

            # ~25k activations: same-bank alternating hammer, one dram.issue.
            obs = self._issue(env, [{"op": "HAMMER", "rows": [a, same_bank], "pairs": 12_500}])
            self.assertIsNone(obs.error)

            # Trace stays within the documented bound despite 25k primitives.
            self.assertLessEqual(len(obs.feedback["trace_tail"]), ISSUE_TRACE_TAIL_CAP)

            # The digest's ACT count reconciles exactly with the worker's cumulative
            # ACT counter — nothing fabricated, nothing double-counted.
            acts_after = int(obs.public_counters["acts"])
            self.assertEqual(obs.feedback["timing_digest"]["acts_delta"], acts_after - acts_before)
            self.assertGreater(obs.feedback["timing_digest"]["acts_delta"], 0)
        finally:
            env.close()

    def test_per_addr_key_is_handle_id_not_hidden_linear(self) -> None:
        # Candidate handles resolve to hidden linear addresses; the per-address
        # digest key must be the handle id (public) — never the linear address.
        env = RowHammerTaskEnv(task=DISCOVERY_TASK)
        try:
            obs = env.reset(seed=7)
            handle = obs.metadata["candidates"][0]["id"]
            hidden_linears = {
                str(env._compiled.target_addr + off)
                for off in (-env._compiled.row_bytes, env._compiled.row_bytes, 2 * env._compiled.row_bytes)
            }
            step = self._issue(env, [{"op": "RD", "addr": {"kind": "handle", "id": handle}}])
            keys = set(step.feedback["timing_digest"]["per_addr_hits"])
            self.assertEqual(keys, {f"handle:{handle}"})
            self.assertFalse(keys & hidden_linears)
        finally:
            env.close()

    def test_digest_absent_when_feedback_not_disclosed(self) -> None:
        # hidden_target ships summarized_counts: no trace, and therefore no digest.
        env = RowHammerTaskEnv(task={"family": "hidden_target"})
        try:
            env.reset(seed=7)
            obs = self._issue(env, [self._rd(0)])
            self.assertEqual(obs.feedback.get("trace_tail"), [])
            self.assertNotIn("timing_digest", obs.feedback)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
