"""Bounded, training-only probe reward shaping (P28).

The discovery reward is sparse (a real flip => 1.0, else 0.0). ``rowhammer_env.llm.
shaping`` adds one auxiliary term to make it learnable: a small dense bonus for each
turn on which the policy made a *decisive* bank-conflict measurement, derived only from
the trusted ``timing_digest``. These tests pin the SPEC §9 contract the term must obey:

* it fires only on a genuine pairwise, repeated same/different-bank probe (not a warm-up,
  not the multi-row confirmation hammer, not a hammer-everything control);
* it is **outcome-neutral** — a clean different-bank reading earns exactly what a clean
  same-bank reading does, so it hints nothing about which candidate is the aggressor;
* it is **bounded in [0,1]** and applied with a weight strictly below ``success_weight``,
  so probing can never earn as much as one real flip (``validate_shaping_weight``);
* it **vanishes from scoring** unless explicitly enabled, and never turns a no-flip
  trajectory into a success — the trusted episode reward stays the only path to 1.0.

All host-runnable with no ``torch``; the integration case is worker-gated.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

from rowhammer_env import RowHammerTaskEnv
from rowhammer_env.llm import ReferenceProbePolicy
from rowhammer_env.llm.multiturn_rollout import ToolPolicyGenerator, run_training_episode_local
from rowhammer_env.llm.shaping import (
    PER_PROBE_BONUS,
    count_decisive_probes,
    is_decisive_probe,
    probe_shaping_reward,
    validate_shaping_weight,
)
from rowhammer_env.observability.metrics import TrajectoryStep

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"


def _digest_step(*, addrs, hits, acts_delta, tool="dram.issue"):
    """A trajectory step carrying a synthetic-but-realistically-shaped timing digest.

    ``hits`` is the per-row row-hit RD count (== read count, §0.2.1); a warm reads each
    row once (hits=1), a repeated probe many times. ``acts_delta`` is the bank-conflict
    discriminator: 0 => different bank (rows stayed open), large => same bank.
    """
    per_addr = {a: {"hits": hits, "misses": acts_delta // max(1, len(addrs)), "acts": 0} for a in addrs}
    return TrajectoryStep(
        action={"tool": tool, "args": {}},
        reward=0.0,
        done=False,
        error=None,
        cycle=0,
        feedback={"timing_digest": {"acts_delta": acts_delta, "per_addr_hits": per_addr}},
    )


class _Rollout:
    def __init__(self, trajectory, reward):
        self.trajectory = trajectory
        self.reward = reward


SAME_BANK = _digest_step(addrs=["cand", "victim"], hits=8, acts_delta=16)
DIFF_BANK = _digest_step(addrs=["cand", "victim"], hits=8, acts_delta=0)
WARM = _digest_step(addrs=["cand", "victim"], hits=1, acts_delta=2)
AMBIGUOUS = _digest_step(addrs=["cand", "victim"], hits=8, acts_delta=1)
MULTI_ROW = _digest_step(addrs=["a", "b", "c"], hits=2700, acts_delta=8100)
SINGLE_ROW = _digest_step(addrs=["victim"], hits=8, acts_delta=8)
NO_DIGEST = TrajectoryStep(action={"tool": "dram.issue", "args": {}}, reward=0.0, done=False, error=None, cycle=0)
FINISH = TrajectoryStep(action={"tool": "episode.finish", "args": {}}, reward=0.0, done=False, error=None, cycle=0)


class DecisiveProbeTests(unittest.TestCase):
    def test_clean_same_bank_probe_is_decisive(self) -> None:
        self.assertTrue(is_decisive_probe(SAME_BANK))

    def test_clean_different_bank_probe_is_decisive(self) -> None:
        self.assertTrue(is_decisive_probe(DIFF_BANK))

    def test_warm_up_is_not_decisive(self) -> None:
        # A one-shot warm reads each row once and opens both regardless of bank — no
        # bank-conflict signal, must not count.
        self.assertFalse(is_decisive_probe(WARM))

    def test_ambiguous_acts_delta_is_not_decisive(self) -> None:
        # A tiny nonzero acts_delta separates neither regime cleanly.
        self.assertFalse(is_decisive_probe(AMBIGUOUS))

    def test_multi_row_hammer_is_not_a_probe(self) -> None:
        # The confirmation hammer / hammer-everything control accesses >2 rows: it is
        # the exploit, not a pairwise same/different-bank measurement.
        self.assertFalse(is_decisive_probe(MULTI_ROW))

    def test_single_row_access_is_not_a_probe(self) -> None:
        self.assertFalse(is_decisive_probe(SINGLE_ROW))

    def test_missing_digest_and_non_issue_are_not_probes(self) -> None:
        self.assertFalse(is_decisive_probe(NO_DIGEST))
        self.assertFalse(is_decisive_probe(FINISH))


class ShapingRewardTests(unittest.TestCase):
    def test_outcome_neutral_same_and_different_bank_earn_equally(self) -> None:
        # The core SPEC §9 property: the bonus rewards making a decisive measurement,
        # NOT which candidate is the aggressor. Swapping every same-bank reading for a
        # different-bank one (and vice versa) must not change the reward.
        traj_a = [WARM, SAME_BANK, WARM, DIFF_BANK]
        traj_b = [WARM, DIFF_BANK, WARM, SAME_BANK]
        self.assertEqual(count_decisive_probes(traj_a), 2)
        self.assertEqual(probe_shaping_reward(_Rollout(traj_a, 0.0)), probe_shaping_reward(_Rollout(traj_b, 0.0)))

    def test_reward_is_bounded_in_unit_interval(self) -> None:
        many = [SAME_BANK] * 100
        self.assertLessEqual(probe_shaping_reward(_Rollout(many, 0.0)), 1.0)
        self.assertEqual(probe_shaping_reward(_Rollout(many, 0.0)), 1.0)  # saturates

    def test_no_probes_earns_nothing(self) -> None:
        self.assertEqual(probe_shaping_reward(_Rollout([WARM, MULTI_ROW, FINISH], 0.0)), 0.0)
        self.assertEqual(probe_shaping_reward(_Rollout([], 0.0)), 0.0)

    def test_scales_with_decisive_probe_count(self) -> None:
        self.assertAlmostEqual(probe_shaping_reward(_Rollout([SAME_BANK, DIFF_BANK], 0.0)), 2 * PER_PROBE_BONUS)

    def test_accepts_a_bare_trajectory_list(self) -> None:
        self.assertAlmostEqual(probe_shaping_reward([SAME_BANK, DIFF_BANK]), 2 * PER_PROBE_BONUS)


class ShapingEnvelopeTests(unittest.TestCase):
    """The term must never masquerade as, or rival, a real success (SPEC §9)."""

    def test_weight_must_be_below_success(self) -> None:
        validate_shaping_weight(1.0, 0.2)  # ok
        validate_shaping_weight(1.0, 0.999)  # ok, still below
        with self.assertRaises(ValueError):
            validate_shaping_weight(1.0, 1.0)  # could equal a real flip
        with self.assertRaises(ValueError):
            validate_shaping_weight(1.0, 1.5)
        with self.assertRaises(ValueError):
            validate_shaping_weight(1.0, -0.1)

    def test_shaping_is_opt_in_and_off_by_default(self) -> None:
        # Structural "vanishes from eval/benchmark scoring": the shaping weight defaults
        # to 0.0, so any config that doesn't explicitly ask for it (the base training
        # config, and every eval/benchmark path, which score on trusted success only)
        # runs with no shaping term at all. Only the discovery *training* curriculum
        # opts in — and even then strictly below success_weight.
        base = yaml.safe_load((ROOT / "configs/training/grpo_qwen8b.yaml").read_text())
        base_reward = base.get("reward", {})
        self.assertEqual(float(base_reward.get("probe_shaping_weight", 0.0)), 0.0)

        curriculum = yaml.safe_load((ROOT / "configs/training/grpo_curriculum.yaml").read_text())
        cur_reward = curriculum["reward"]
        weight = float(cur_reward["probe_shaping_weight"])
        self.assertGreater(weight, 0.0)  # training opts in
        validate_shaping_weight(float(cur_reward.get("success_weight", 1.0)), weight)  # still below success

    def test_probing_never_reaches_a_real_success(self) -> None:
        # A no-flip trajectory with the maximum possible shaping still scores strictly
        # below what any real flip earns — for every admissible weight. The trusted
        # reward (0.0 here) is untouched by shaping; shaping is purely additive and
        # bounded, so it can never turn "no flip" into a success.
        success_weight, shaping_weight = 1.0, 0.2
        validate_shaping_weight(success_weight, shaping_weight)
        no_flip = _Rollout([SAME_BANK] * 100, reward=0.0)  # perfect probing, no flip
        flip_no_probes = _Rollout([FINISH], reward=1.0)  # a real flip, zero shaping
        max_no_flip = success_weight * no_flip.reward + shaping_weight * probe_shaping_reward(no_flip)
        min_flip = success_weight * flip_no_probes.reward + shaping_weight * probe_shaping_reward(flip_no_probes)
        self.assertEqual(no_flip.reward, 0.0)  # shaping did not create trusted success
        self.assertLess(max_no_flip, min_flip)
        self.assertLess(max_no_flip, success_weight)  # never as much as one real flip


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class ShapingIntegrationTests(unittest.TestCase):
    def test_reference_rollout_earns_shaping_from_real_digests(self) -> None:
        # On a real worker rollout, the deterministic reference policy issues one pairwise
        # timing probe per candidate; each produces a real, decisive digest, so shaping is
        # positive and saturates — driven entirely by trusted timing state.
        for family in ("bounded_sweep", "hidden_adjacency"):
            task = yaml.safe_load((ROOT / f"configs/tasks/{family}_easy.yaml").read_text())
            env = RowHammerTaskEnv(task=task)
            try:
                rollout = run_training_episode_local(
                    env, ToolPolicyGenerator(ReferenceProbePolicy()), seed=2, task=task, max_turns=96
                )
            finally:
                env.close()
            self.assertEqual(rollout.reward, 1.0, family)
            self.assertGreaterEqual(count_decisive_probes(rollout.trajectory), 4, family)
            self.assertEqual(probe_shaping_reward(rollout), 1.0, family)


if __name__ == "__main__":
    unittest.main()
