"""Phase 13 — task compiler + families (defect J; TEST_PLAN E3/E7/E6).

Engine-level and compiler-level checks that run in the source-free unit gate (they
construct the real ``DisturbanceEngine`` from the admitted profile, no worker
build required). The end-to-end worker path is covered by
``scripts/verify_phase13.py``; the worker-gated class below adds a fast smoke test
when the binary is present.
"""

from __future__ import annotations

import json
import pathlib
import unittest

import jsonschema
import yaml

from rowhammer_env import Phase2Action, RowHammerTaskEnv
from rowhammer_env.disturbance import DisturbanceEngine
from rowhammer_env.geometry import Geometry
from rowhammer_env.rewards import success_for
from rowhammer_env.tasks import Disclosure
from rowhammer_env.tasks.compiler import (
    BAND_ACTS,
    FAMILIES,
    LEGACY_BUDGETS,
    TaskConfigError,
    TaskSpec,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
TASK_DIR = ROOT / "configs/tasks"
SCHEMA = json.loads((ROOT / "spec/schemas/task.schema.json").read_text())

DDR4_INFO = {
    "standard": "DDR4",
    "tx_bytes": 64,
    "prefetch": 8,
    "channel_width": 64,
    "level_names": ["Channel", "Rank", "BankGroup", "Bank", "Row", "Column"],
    "level_sizes": [1, 1, 4, 4, 65536, 1024],
}


def geo() -> Geometry:
    return Geometry(DDR4_INFO)


def compile_task(config, seed=13):
    return TaskSpec.from_config(config).compile(seed, geo())


class CompilerParsingTests(unittest.TestCase):
    def test_families_registered(self) -> None:
        # The ten SPEC §7 core families, plus the Tier 2a ``bounded_sweep`` (P23)
        # and Tier 2b ``hidden_adjacency`` (P25) discovery families added by
        # IMPLEMENTATION_PLAN_V3.
        spec_families = {
            "known_target_anybit", "target_row", "target_cell", "pattern_target", "any_flip",
            "hidden_target", "unknown_adjacency", "mitigation_aware", "low_disclosure",
            "profile_generalization",
        }
        self.assertTrue(spec_families <= set(FAMILIES))
        self.assertIn("bounded_sweep", FAMILIES)
        self.assertIn("hidden_adjacency", FAMILIES)
        self.assertEqual(len(FAMILIES), 12)

    def test_legacy_shorthand_and_alias(self) -> None:
        spec = TaskSpec.from_config({"family": "known_target"})  # alias
        self.assertEqual(spec.family, "known_target_anybit")
        self.assertEqual(spec.disclosure.mapping, "physical")

    def test_bare_config_defaults_to_known_target(self) -> None:
        self.assertEqual(TaskSpec.from_config(None).family, "known_target_anybit")
        self.assertEqual(TaskSpec.from_config({}).family, "known_target_anybit")

    def test_family_derived_from_full_config(self) -> None:
        cfg = {
            "objective": {"type": "target_row_flip"},
            "disclosure": {"mapping": "logical_only", "adjacency": "candidate_set",
                           "victim": "row_handle", "profile": "public_profile_id", "feedback": "summarized_counts"},
        }
        self.assertEqual(TaskSpec.from_config(cfg).family, "unknown_adjacency")
        anyflip = {"objective": {"type": "any_flip"}, "disclosure": Disclosure().as_public()}
        self.assertEqual(TaskSpec.from_config(anyflip).family, "any_flip")

    def test_unknown_family_and_band_fail_closed(self) -> None:
        with self.assertRaises(TaskConfigError):
            TaskSpec.from_config({"family": "does_not_exist"})
        with self.assertRaises(TaskConfigError):
            TaskSpec.from_config({"family": "any_flip", "difficulty": "impossible"})

    def test_explicit_mitigation_and_budgets(self) -> None:
        spec = TaskSpec.from_config({"family": "mitigation_aware", "mitigation": {"name": "oracle", "params": {"tRH": 42}}})
        self.assertEqual(spec.mitigation, {"name": "oracle", "params": {"tRH": 42}})

    def test_graded_budget_from_band(self) -> None:
        spec = TaskSpec.from_config({"family": "any_flip", "difficulty": "hard"})
        self.assertEqual(spec.resolved_budgets()["acts"], BAND_ACTS["hard"])
        # Non-graded families keep the unbounded legacy budgets.
        self.assertEqual(TaskSpec.from_config({"family": "known_target_anybit"}).resolved_budgets(), dict(LEGACY_BUDGETS))


class CompilerSamplingTests(unittest.TestCase):
    def test_target_sampling_is_deterministic_per_seed(self) -> None:
        a = compile_task({"family": "known_target_anybit"}, seed=7)
        b = compile_task({"family": "known_target_anybit"}, seed=7)
        self.assertEqual((a.target_row, a.target_bit, a.target_addr), (b.target_row, b.target_bit, b.target_addr))

    def test_seed_changes_target(self) -> None:
        rows = {compile_task({"family": "known_target_anybit"}, seed=s).target_row for s in range(8)}
        self.assertGreater(len(rows), 1)  # not the old hardcoded single row

    def test_known_kind_routes_target_through_engine_known_row(self) -> None:
        ct = compile_task({"family": "known_target_anybit"}, seed=3)
        self.assertEqual(ct.target_kind, "known")
        self.assertEqual(ct.engine_known_row, ct.target_row)  # gets the fixed calibrated hcfirst
        self.assertGreater(ct.target_row, 10)

    def test_sampled_kind_keeps_reserved_known_row(self) -> None:
        ct = compile_task({"family": "any_flip"}, seed=3)
        self.assertEqual(ct.target_kind, "sampled")
        self.assertEqual(ct.engine_known_row, 10)  # target is a separately sampled, profile-thresholded row
        self.assertGreater(ct.target_row, 1000)

    def test_disturbance_overrides(self) -> None:
        cell = compile_task({"family": "target_cell", "bit": 5}, seed=1)
        self.assertEqual(cell.disturbance_overrides()["known_first_bit"], 5)
        pg = compile_task({"family": "profile_generalization", "split": "eval"}, seed=1)
        self.assertEqual(pg.disturbance_overrides()["family"], "axmicr")


class RewardPredicateTests(unittest.TestCase):
    """Each family predicate reads only trusted flips (SPEC §9)."""

    def _engine_for(self, ct) -> DisturbanceEngine:
        return DisturbanceEngine(geometry=geo(), seed=ct.seed, **ct.disturbance_overrides())

    def test_none_inputs_are_false(self) -> None:
        ct = compile_task({"family": "any_flip"})
        self.assertFalse(success_for(None, self._engine_for(ct)))
        self.assertFalse(success_for(ct, None))

    def test_target_row_predicate(self) -> None:
        ct = compile_task({"family": "known_target_anybit"}, seed=4)
        eng = self._engine_for(ct)
        self.assertFalse(success_for(ct, eng))
        eng.flips[ct.target_addr + 3] = 0  # a cell in the target row
        self.assertTrue(success_for(ct, eng))

    def test_any_flip_predicate(self) -> None:
        ct = compile_task({"family": "any_flip"}, seed=4)
        eng = self._engine_for(ct)
        self.assertFalse(success_for(ct, eng))
        eng.flips[999 * eng.row_bytes] = 2  # any cell anywhere
        self.assertTrue(success_for(ct, eng))

    def test_target_cell_predicate_requires_exact_bit(self) -> None:
        ct = compile_task({"family": "target_cell", "bit": 6}, seed=4)
        eng = self._engine_for(ct)
        eng.flips[ct.target_addr] = 3  # wrong bit
        self.assertFalse(success_for(ct, eng))
        eng.flips[ct.target_addr] = 6  # requested bit
        self.assertTrue(success_for(ct, eng))

    def test_pattern_predicate(self) -> None:
        ct = compile_task({"family": "pattern_target", "bit": 0}, seed=4)
        eng = self._engine_for(ct)
        self.assertFalse(success_for(ct, eng))
        eng.flips[ct.target_addr] = ct.target_bit
        self.assertTrue(success_for(ct, eng))


class TaskConfigFileTests(unittest.TestCase):
    def test_all_configs_validate_and_compile_and_cover_families(self) -> None:
        seen = set()
        configs = sorted(TASK_DIR.glob("*.yaml"))
        self.assertTrue(configs)
        for path in configs:
            doc = yaml.safe_load(path.read_text())
            jsonschema.validate(doc, SCHEMA)
            spec = TaskSpec.from_config(doc)
            ct = spec.compile(seed=1, geometry=geo())
            self.assertEqual(ct.family, spec.family)
            self.assertIn("acts", ct.budgets)
            seen.add(spec.family)
        self.assertEqual(seen, set(FAMILIES))

    def test_spec_example_still_validates(self) -> None:
        example = json.loads((ROOT / "spec/examples/task.known_target.json").read_text())
        jsonschema.validate(example, SCHEMA)


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class TaskEnvIntegrationTests(unittest.TestCase):
    def _rd(self, addr):
        return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}

    def test_known_target_flips_and_is_reproducible(self) -> None:
        outcomes = []
        for _ in range(2):
            env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
            obs = env.reset(seed=21)
            d = env.disturbance
            target, rb, thr = d.target_addr, d.row_bytes, d.known_threshold
            last = None
            for _ in range(thr // 2):
                last = env.step(Phase2Action(tool="dram.issue", args={"commands": [self._rd(target - rb), self._rd(target + rb)]}))
                if last.done:
                    break
            outcomes.append((last.reward, last.done, obs.metadata["target"]["addr"]))
            env.close()
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][0], 1.0)

    def test_unknown_adjacency_solvable_via_candidate_handles(self) -> None:
        env = RowHammerTaskEnv(task={"family": "unknown_adjacency"})
        obs = env.reset(seed=21)
        cands = [c["id"] for c in obs.metadata["candidates"]]
        thr = env.disturbance.known_threshold
        last = None
        for _ in range(thr // 2):
            last = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "RD", "addr": {"kind": "handle", "id": cands[0]}},
                             {"op": "RD", "addr": {"kind": "handle", "id": cands[1]}}]}))
            if last.done:
                break
        self.assertEqual(last.reward, 1.0)
        env.close()

    def test_hidden_target_finish_is_zero_and_hides_target(self) -> None:
        env = RowHammerTaskEnv(task={"family": "hidden_target"})
        obs = env.reset(seed=21)
        self.assertEqual(obs.metadata["target"]["kind"], "handle")
        self.assertNotIn("known_target_row", obs.metadata["disturbance"])
        fin = env.step(Phase2Action(tool="episode.finish", args={}))
        self.assertTrue(fin.done)
        self.assertEqual(fin.reward, 0.0)

    def test_acts_budget_exhaustion_fails_closed(self) -> None:
        env = RowHammerTaskEnv(task={"family": "any_flip", "difficulty": "hard", "id": "ddr4_unit_hard_v1"})
        env.reset(seed=99)
        rb = env._compiled.row_bytes
        vaddr = env._compiled.target_row * rb
        last = None
        for _ in range(100000):
            last = env.step(Phase2Action(tool="dram.issue", args={"commands": [self._rd(vaddr - rb), self._rd(vaddr + rb)]}))
            if last.done:
                break
        self.assertTrue(last.done)
        # Either it flipped within the hard budget or it failed closed on acts.
        if last.reward == 0.0:
            self.assertEqual(last.error["code"], "BUDGET_EXCEEDED")
        env.close()


if __name__ == "__main__":
    unittest.main()
