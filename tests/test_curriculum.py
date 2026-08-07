"""Discovery curriculum ordering + the P26 reference gate (P28).

``rowhammer_env.llm.curriculum`` is the single source of truth for the training
curriculum: an ordered set of stages, easiest first, each gated by the fraction of
episodes the deterministic P26 reference policy must solve within budget before the
stage earns instance time. These tests pin two things:

* **Structure (host, no worker).** The shipped ``grpo_curriculum.yaml`` parses into the
  intended Tier 0 → Tier 2a easy/med/hard → Tier 2b easy/med/hard order, the parser
  rejects a mis-ordered or malformed curriculum, and the flattening preserves order.
* **The gate (worker-gated).** For every stage, the appropriate reference policy —
  driven through the *actual training rollout* (``run_training_episode_local``) — clears
  the stage's ``reference_min_success``. This is P28.3: re-validate the curriculum on the
  host before spending any GPU time; a band the deterministic reference can't solve,
  GRPO won't either.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

from rowhammer_env import RowHammerTaskEnv
from rowhammer_env.llm import CIHammerFixturePolicy, ReferenceProbePolicy
from rowhammer_env.llm.curriculum import (
    BAND_ORDER,
    TIER_ORDER,
    CurriculumStage,
    curriculum_task_seed_pairs,
    load_curriculum,
)
from rowhammer_env.llm.multiturn_rollout import ToolPolicyGenerator, run_training_episode_local

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
CURRICULUM_CONFIG = ROOT / "configs/training/grpo_curriculum.yaml"


def _cfg() -> dict:
    return yaml.safe_load(CURRICULUM_CONFIG.read_text())


def _budget_exceeded(result) -> bool:
    return any((step.error or {}).get("code") == "BUDGET_EXCEEDED" for step in result.trajectory)


class CurriculumStructureTests(unittest.TestCase):
    def test_shipped_curriculum_is_the_intended_ordering(self) -> None:
        stages = load_curriculum(_cfg())
        got = [(s.tier, s.band) for s in stages]
        self.assertEqual(
            got,
            [
                ("tier0", ""),
                ("tier2a", "easy"),
                ("tier2a", "medium"),
                ("tier2a", "hard"),
                ("tier2b", "easy"),
                ("tier2b", "medium"),
                ("tier2b", "hard"),
            ],
        )
        # Every stage names real, existing task configs and a valid gate window.
        for stage in stages:
            self.assertTrue(stage.tasks)
            for task in stage.tasks:
                self.assertTrue((ROOT / task).is_file(), task)
            self.assertTrue(0.0 <= stage.reference_min_success <= 1.0)
            self.assertTrue(stage.seeds)

    def test_difficulty_is_non_decreasing(self) -> None:
        stages = load_curriculum(_cfg())
        ranks = [(TIER_ORDER.index(s.tier), BAND_ORDER.index(s.band)) for s in stages]
        self.assertEqual(ranks, sorted(ranks))

    def test_parser_rejects_out_of_order_curriculum(self) -> None:
        # Tier 2b before Tier 2a is a curriculum bug the parser must catch, not run.
        bad = {
            "curriculum": [
                {"name": "b", "tier": "tier2b", "band": "easy", "tasks": ["x.yaml"], "seeds": [1], "reference_min_success": 1.0},
                {"name": "a", "tier": "tier2a", "band": "easy", "tasks": ["y.yaml"], "seeds": [1], "reference_min_success": 1.0},
            ]
        }
        with self.assertRaises(ValueError):
            load_curriculum(bad)

    def test_parser_rejects_malformed_stage(self) -> None:
        with self.assertRaises(ValueError):
            load_curriculum({"curriculum": [{"name": "x", "tier": "tier0", "tasks": [], "seeds": [1], "reference_min_success": 1.0}]})
        with self.assertRaises(ValueError):  # missing the P26 gate
            load_curriculum({"curriculum": [{"name": "x", "tier": "tier0", "tasks": ["a.yaml"], "seeds": [1]}]})
        with self.assertRaises(ValueError):  # unknown tier
            load_curriculum({"curriculum": [{"name": "x", "tier": "tierX", "tasks": ["a.yaml"], "seeds": [1], "reference_min_success": 1.0}]})
        with self.assertRaises(ValueError):  # no curriculum at all
            load_curriculum({})

    def test_flattening_preserves_stage_then_task_then_seed_order(self) -> None:
        stages = [
            CurriculumStage("s0", "tier0", "", ("a.yaml",), (1, 2), True, 1.0),
            CurriculumStage("s1", "tier2a", "easy", ("b.yaml",), (7,), True, 1.0),
        ]
        self.assertEqual(
            curriculum_task_seed_pairs(stages),
            [("a.yaml", 1), ("a.yaml", 2), ("b.yaml", 7)],
        )

    def test_seed_range_shorthand_is_inclusive(self) -> None:
        # [1, 4] expands to seeds 1,2,3,4 (both ends inclusive); an explicit list is
        # honored verbatim.
        base = {"name": "x", "tier": "tier0", "band": "", "tasks": ["a.yaml"], "reference_min_success": 1.0}
        (ranged,) = load_curriculum({"curriculum": [{**base, "seed_range": [1, 4]}]})
        self.assertEqual(ranged.seeds, (1, 2, 3, 4))
        (single,) = load_curriculum({"curriculum": [{**base, "seed_range": [10, 10]}]})
        self.assertEqual(single.seeds, (10,))
        (listed,) = load_curriculum({"curriculum": [{**base, "seeds": [3, 1, 2]}]})
        self.assertEqual(listed.seeds, (3, 1, 2))

    def test_seed_range_rejects_bad_forms(self) -> None:
        base = {"name": "x", "tier": "tier0", "band": "", "tasks": ["a.yaml"], "reference_min_success": 1.0}
        with self.assertRaises(ValueError):  # start > stop
            load_curriculum({"curriculum": [{**base, "seed_range": [9, 1]}]})
        with self.assertRaises(ValueError):  # not a [start, stop] pair
            load_curriculum({"curriculum": [{**base, "seed_range": [1, 2, 3]}]})
        with self.assertRaises(ValueError):  # both forms at once
            load_curriculum({"curriculum": [{**base, "seeds": [1], "seed_range": [1, 4]}]})
        with self.assertRaises(ValueError):  # neither form
            load_curriculum({"curriculum": [{**base}]})


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class CurriculumReferenceGateTests(unittest.TestCase):
    """P28.3 / P28 Done-when: the reference policy hits every band's window.

    Driven through the real training rollout (``run_training_episode_local``), the same
    path GRPO uses. Tier 0 is solved by the disclosed-hint hammer fixture; the discovery
    tiers by the deterministic DRAMA reference. A stage passes iff the solved fraction
    over its seeds meets ``reference_min_success`` with no budget overrun.
    """

    # Cap seeds per stage so the gate stays fast; the wider seed/band sweep lives in
    # tests/test_reference_policy.py. reference_min_success is 1.0, so each checked seed
    # must solve within budget.
    SEEDS_PER_STAGE = 2

    def _policy_for(self, stage: CurriculumStage):
        # Tier 0 discloses the exact aggressors + threshold -> the reference hammer
        # fixture saturates it. The discovery tiers need the DRAMA timing solver.
        return CIHammerFixturePolicy() if stage.tier == "tier0" else ReferenceProbePolicy()

    def test_reference_policy_clears_every_stage_window(self) -> None:
        for stage in load_curriculum(_cfg()):
            seeds = stage.seeds[: self.SEEDS_PER_STAGE]
            for task_path in stage.tasks:
                task = yaml.safe_load((ROOT / task_path).read_text())
                solved = 0
                for seed in seeds:
                    env = RowHammerTaskEnv(task=task)
                    try:
                        result = run_training_episode_local(
                            env, ToolPolicyGenerator(self._policy_for(stage)), seed=seed, task=task, max_turns=200
                        )
                    finally:
                        env.close()
                    self.assertFalse(_budget_exceeded(result), (stage.name, seed))
                    if result.reward >= 1.0:
                        solved += 1
                rate = solved / len(seeds)
                self.assertGreaterEqual(
                    rate, stage.reference_min_success, f"{stage.name}: solved {rate} < window {stage.reference_min_success}"
                )


if __name__ == "__main__":
    unittest.main()
