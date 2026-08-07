"""Reference probing policy — the go/no-go before any training (P26).

``ReferenceProbePolicy`` is a deterministic DRAMA solver that proves the disclosed
signals are *sufficient* to solve the Tier 2 discovery families using exactly the
tools an LLM policy has: it classifies each candidate same-bank vs different-bank by
the bank-conflict timing channel (``timing_digest.acts_delta``, never the RD's own
``row_hit`` — always true, §0.2.1), then double-sides the same-bank survivors in one
budgeted ``HAMMER``. It runs on the host with no ``torch``.

Two layers of checks:

* **Unit (no worker).** The policy state machine is driven through
  ``run_episode_local`` against a fake env that implements only the timing/flip
  contract — asserting it probes every candidate, hammers *only* the same-bank
  survivors, and that the timing-blind control (``use_timing=False``) skips probing
  and hammers every candidate.
* **Integration (worker-gated).** The real ``RowHammerTaskEnv`` under the shipped
  band configs: the reference solves ``easy``/``medium`` for both ``bounded_sweep``
  (Tier 2a) and ``hidden_adjacency`` (Tier 2b) across many seeds *within* the
  calibrated activation budget; the same-bank set it recovers by timing matches the
  secret mapper's decode; and a timing-blind control trips ``BUDGET_EXCEEDED`` on the
  bands where the reference stays within budget — proving the signal is load-bearing.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

from rowhammer_env import Phase2Action, RowHammerTaskEnv
from rowhammer_env.llm import ReferenceProbePolicy, run_episode_local
from rowhammer_env.phase2_env import Phase2Observation

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"


def _config(family: str, band: str) -> dict:
    return yaml.safe_load((ROOT / f"configs/tasks/{family}_{band}.yaml").read_text())


def _budget_exceeded(result) -> bool:
    return any((step.error or {}).get("code") == "BUDGET_EXCEEDED" for step in result.trajectory)


# --- Unit: the state machine against a fake timing/flip contract (no worker) -------


class _FakeDiscoveryEnv:
    """Minimal env implementing only the timing digest + double-sided flip contract.

    Candidates and the victim are numeric-address dicts; a preset ``banks`` map gives
    each candidate's bank (the victim is bank 0). A probe (a two-row ``HAMMER`` that
    includes the victim) returns ``acts_delta>0`` iff the other row shares the
    victim's bank. A confirmation ``HAMMER`` (victim absent) flips iff both true
    aggressors are in the hammered rows — exactly the real double-sided condition.
    """

    def __init__(self, victim: dict, candidates: list[dict], banks: dict[int, int], aggressors: set[int]) -> None:
        self._victim = victim
        self._candidates = candidates
        self._banks = banks  # addr -> bank
        self._aggressors = aggressors  # addrs
        self.probed: list[dict] = []
        self.hammered_rows: list[dict] | None = None
        self._flipped = False

    def reset(self, seed=None, episode_id=None, **_) -> Phase2Observation:
        del seed, episode_id
        return Phase2Observation(
            reward=0.0,
            done=False,
            metadata={"objective": {"type": "target_row_flip", "target": self._victim},
                      "candidates": list(self._candidates)},
        )

    def step(self, action: Phase2Action) -> Phase2Observation:
        if action.tool == "episode.finish":
            return Phase2Observation(reward=1.0 if self._flipped else 0.0, done=True)
        if action.tool != "dram.issue":
            return Phase2Observation(reward=0.0, done=False)
        commands = action.args["commands"]
        if all(c["op"] == "RD" for c in commands):  # warm both rows open
            return Phase2Observation(reward=0.0, done=False)
        rows = commands[0]["rows"]
        if self._victim in rows:  # a probe: victim alternated with one candidate
            candidate = next(r for r in rows if r != self._victim)
            self.probed.append(candidate)
            same_bank = self._banks[candidate["addr"]] == 0
            return Phase2Observation(
                reward=0.0, done=False,
                feedback={"timing_digest": {"acts_delta": 16 if same_bank else 0,
                                            "cycles_delta": 4000 if same_bank else 40}},
            )
        # confirmation hammer over the (victim-free) survivor set
        self.hammered_rows = rows
        hammered = {r["addr"] for r in rows}
        if self._aggressors <= hammered:
            self._flipped = True
            return Phase2Observation(reward=1.0, done=True, feedback={"new_public_flips": 1})
        return Phase2Observation(reward=0.0, done=False, feedback={"new_public_flips": 0})

    def close(self) -> None:  # pragma: no cover - parity with the real env
        pass


class ReferenceProbePolicyUnitTests(unittest.TestCase):
    def _window(self):
        victim = {"kind": "logical", "addr": 1000}
        # aggr_a & aggr_b: same bank (0) aggressors; far: same bank non-aggressor; diff: other bank.
        cands = [{"kind": "logical", "addr": a} for a in (10, 20, 30, 40)]
        banks = {10: 0, 20: 1, 30: 0, 40: 0}          # 20 is a different-bank decoy
        aggressors = {10, 40}                          # 30 is a same-bank-far decoy
        same_bank = [c for c in cands if banks[c["addr"]] == 0]
        return victim, cands, banks, aggressors, same_bank

    def test_reference_probes_all_and_hammers_only_same_bank_survivors(self) -> None:
        victim, cands, banks, aggressors, same_bank = self._window()
        env = _FakeDiscoveryEnv(victim, cands, banks, aggressors)
        policy = ReferenceProbePolicy()
        result = run_episode_local(env, policy, seed=0)
        # Every candidate is classified by a timing probe...
        self.assertEqual(env.probed, cands)
        # ...and only the same-bank survivors are hammered (the different-bank decoy
        # is dropped without spending a hammer on it).
        self.assertEqual(env.hammered_rows, same_bank)
        self.assertNotIn({"kind": "logical", "addr": 20}, env.hammered_rows)
        self.assertEqual(result.reward, 1.0)

    def test_timing_blind_control_skips_probes_and_hammers_all(self) -> None:
        victim, cands, banks, aggressors, _ = self._window()
        env = _FakeDiscoveryEnv(victim, cands, banks, aggressors)
        result = run_episode_local(env, ReferenceProbePolicy(use_timing=False), seed=0)
        self.assertEqual(env.probed, [])              # no timing probes at all
        self.assertEqual(env.hammered_rows, cands)    # hammers every candidate
        self.assertEqual(result.reward, 1.0)

    def test_no_candidates_finishes_immediately(self) -> None:
        env = _FakeDiscoveryEnv({"kind": "logical", "addr": 1}, [], {}, set())
        result = run_episode_local(env, ReferenceProbePolicy(), seed=0)
        self.assertEqual(result.trajectory[0].action["tool"], "episode.finish")


# --- Integration: the real worker-gated env under the shipped configs --------------


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class ReferenceProbeIntegrationTests(unittest.TestCase):
    FAMILIES = ("bounded_sweep", "hidden_adjacency")

    def test_reference_solves_easy_and_medium_within_budget(self) -> None:
        # The go/no-go for training: a deterministic policy using only the disclosed
        # tools/signals solves both Tier 2a and Tier 2b across many seeds inside the
        # calibrated activation budget — no BUDGET_EXCEEDED anywhere.
        for family in self.FAMILIES:
            for band in ("easy", "medium"):
                task = _config(family, band)
                for seed in range(4):
                    env = RowHammerTaskEnv(task=task)
                    try:
                        result = run_episode_local(env, ReferenceProbePolicy(), seed=seed, task=task)
                    finally:
                        env.close()
                    self.assertEqual(result.reward, 1.0, (family, band, seed))
                    self.assertFalse(_budget_exceeded(result), (family, band, seed))

    def test_reference_solves_hard_within_budget(self) -> None:
        # Smoke that the widest band is also solvable within its (larger) budget.
        for family in self.FAMILIES:
            task = _config(family, "hard")
            env = RowHammerTaskEnv(task=task)
            try:
                result = run_episode_local(env, ReferenceProbePolicy(), seed=3, task=task)
            finally:
                env.close()
            self.assertEqual(result.reward, 1.0, family)
            self.assertFalse(_budget_exceeded(result), family)

    def test_recovered_same_bank_set_matches_the_secret_mapper(self) -> None:
        # The timing classification is *correct*, not incidental: the same-bank set
        # the policy recovers by timing is exactly the set of candidates that decode
        # (under the per-episode secret mapper) into the victim's bank, and it always
        # contains both true aggressors. Compared by candidate index, so it holds for
        # both handle (Tier 2a) and numeric (Tier 2b) candidates — the disclosed
        # candidate order mirrors ``ct.candidates``.
        for family in self.FAMILIES:
            task = _config(family, "medium")
            env = RowHammerTaskEnv(task=task)
            policy = ReferenceProbePolicy()
            try:
                result = run_episode_local(env, policy, seed=5, task=task)
                ct = env._compiled
                vbank = (ct.target_bankgroup, ct.target_bank)
                recovered = {policy._candidates.index(row) for row in policy._same_bank}
                truth = {
                    i
                    for i, c in enumerate(ct.candidates)
                    if (lambda d: (d["bankgroup"], d["bank"]) == vbank)(env._decode(ct.target_addr + c.offset))
                }
                aggressors = {i for i, c in enumerate(ct.candidates) if c.is_aggressor}
            finally:
                env.close()
            self.assertEqual(recovered, truth, family)
            self.assertTrue(aggressors <= recovered, family)
            self.assertEqual(result.reward, 1.0, family)

    def test_timing_blind_control_fails_where_reference_succeeds(self) -> None:
        # Differential: the timing signal is load-bearing. Under the calibrated
        # budget the timing-narrowed reference solves within the activation budget,
        # while a control that ignores the timing (hammers every candidate) has its
        # over-budget hammer truncated at the budget and *fails* — no flip, reward
        # 0.0, BUDGET_EXCEEDED. Without the timing channel the budget cannot be met.
        for family, band, seed in (("hidden_adjacency", "medium", 1), ("bounded_sweep", "hard", 1)):
            task = _config(family, band)
            env = RowHammerTaskEnv(task=task)
            try:
                reference = run_episode_local(env, ReferenceProbePolicy(), seed=seed, task=task)
            finally:
                env.close()
            env = RowHammerTaskEnv(task=task)
            try:
                control = run_episode_local(env, ReferenceProbePolicy(use_timing=False), seed=seed, task=task)
            finally:
                env.close()
            self.assertEqual(reference.reward, 1.0, (family, band))
            self.assertFalse(_budget_exceeded(reference), (family, band))
            self.assertEqual(control.reward, 0.0, (family, band))
            self.assertTrue(_budget_exceeded(control), (family, band))


if __name__ == "__main__":
    unittest.main()
