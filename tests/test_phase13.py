"""Phase 13 — task compiler + families (defect J; TEST_PLAN E3/E7/E6).

Engine-level and compiler-level checks that run in the source-free unit gate (they
construct the real ``DisturbanceEngine`` from the admitted profile, no worker
build required). The end-to-end worker path is covered by
``scripts/verify_phase13.py``; the worker-gated class below adds a fast smoke test
when the binary is present.
"""

from __future__ import annotations

import base64
import json
import pathlib
import unittest

import jsonschema
import yaml

from rowhammer_env import Phase2Action, RowHammerTaskEnv
from rowhammer_env.disturbance import DisturbanceEngine
from rowhammer_env.geometry import Geometry
from rowhammer_env.tools.addressing import AddressMapper
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


def _column0_addr(eng: DisturbanceEngine, row: int, bank: int) -> int:
    """Linear address of column 0 of ``(bank, row)`` under RoBaRaCoCh, bankgroup 0."""
    return row * eng.row_bytes + bank * eng.row_span * int(eng.geometry.level_sizes["bankgroup"])


def _act(row: int, bank: int, clk: int) -> dict:
    return {"op": "ACT", "channel": 0, "rank": 0, "bankgroup": 0, "bank": bank,
            "row": row, "row_hit": False, "clk": clk}


def _rd(addr: int) -> dict:
    return {"op": "RD", "addr": addr, "size": 64}


def _unexpected_read(addr: int) -> int:
    raise AssertionError(f"predicate unexpectedly read byte {addr}")


def _stored_byte_reader(eng: DisturbanceEngine, stored: int):
    """Model the production worker-stored-byte + disturbance.apply boundary."""
    return lambda addr: eng.apply(addr, bytes([stored]))[0]


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

    def test_public_mapper_victim_key_is_decoded_and_pins_the_known_threshold(self) -> None:
        # The row-aligned victim address really does decode to channel/rank/
        # bankgroup/bank 0 under RoBaRaCoCh — but it is *decoded*, not assumed, and
        # the engine pins the fixed threshold to the same key the predicate reads.
        ct = compile_task({"family": "known_target_anybit"}, seed=4)
        self.assertEqual(ct.target_row_key, (0, 0, 0, 0, ct.target_row))
        overrides = ct.disturbance_overrides()
        pinned = (
            overrides["known_target_channel"],
            overrides["known_target_rank"],
            overrides["known_target_bankgroup"],
            overrides["known_target_bank"],
            overrides["known_target_row"],
        )
        self.assertEqual(pinned, ct.target_row_key)

    def test_decoded_victim_coordinates_are_carried_through(self) -> None:
        # Whatever the trusted decode reports is what the key holds — a secret
        # mapper scrambles the bank, and a multi-rank part can place the victim off
        # rank 0. Neither may be silently replaced by 0.
        decoded = {"channel": 0, "rank": 1, "bankgroup": 3, "bank": 2, "row": 0, "column": 0}
        ct = TaskSpec.from_config({"family": "known_target_anybit"}).compile(
            seed=4, geometry=geo(), decode=lambda addr: decoded
        )
        self.assertEqual(ct.target_row_key, (0, 1, 3, 2, ct.target_row))
        self.assertEqual(ct.disturbance_overrides()["known_target_rank"], 1)

    def test_undecodable_geometry_fails_closed(self) -> None:
        # No projection for multi-channel (@spec:sim-not-modeled): compiling must
        # raise rather than emit a victim key that can never match a flip.
        multi_channel = dict(DDR4_INFO, level_sizes=[2, 1, 4, 4, 65536, 1024])
        with self.assertRaises(TaskConfigError):
            TaskSpec.from_config({"family": "known_target_anybit"}).compile(
                seed=1, geometry=Geometry(multi_channel)
            )


class RewardPredicateTests(unittest.TestCase):
    """Each family predicate reads only trusted flips (SPEC §9)."""

    def _engine_for(self, ct) -> DisturbanceEngine:
        return DisturbanceEngine(
            geometry=geo(), row_encoder=AddressMapper(geo()).encode, seed=ct.seed, **ct.disturbance_overrides()
        )

    def test_none_inputs_are_false(self) -> None:
        ct = compile_task({"family": "any_flip"})
        self.assertFalse(success_for(None, self._engine_for(ct), _unexpected_read))
        self.assertFalse(success_for(ct, None, _unexpected_read))

    def test_target_row_predicate_reads_the_decoded_victim_key(self) -> None:
        # @spec:rl-reward — "the target row" is a full physical coordinate, so the
        # row *index* alone (shared by one row per bank) must not decide success.
        ct = compile_task({"family": "known_target_anybit"}, seed=4)
        eng = self._engine_for(ct)
        self.assertFalse(success_for(ct, eng, _unexpected_read))
        eng.flipped_row_keys.add((0, 0, 3, 2, ct.target_row))  # same index, other bank
        self.assertFalse(success_for(ct, eng, _unexpected_read))
        eng.flipped_row_keys.add(ct.target_row_key)
        self.assertTrue(success_for(ct, eng, _unexpected_read))

    def test_a_flip_earned_in_another_bank_does_not_win_target_cell(self) -> None:
        # @spec:rl-reward — the cell predicates key on ``target_addr`` (the victim
        # row's column 0), so the victim anchor is what keeps them honest: anchoring
        # on the row *stride* instead of the column field drops the aggressor's bank
        # and files a wrong-bank flip at exactly ``target_addr``. Driven end to end
        # against the real engine because the anchor, not the predicate, is the part
        # that can regress.
        ct = compile_task({"family": "target_cell", "id": "tc"}, seed=13)
        eng = self._engine_for(ct)
        bank = 2
        lo = _column0_addr(eng, ct.target_row - 1, bank)
        hi = _column0_addr(eng, ct.target_row + 1, bank)
        wrong_bank_key = (0, 0, 0, bank, ct.target_row)
        for i in range(200_000):
            eng.consume([_act(ct.target_row - 1, bank, 2 * i)], _rd(lo))
            eng.consume([_act(ct.target_row + 1, bank, 2 * i + 1)], _rd(hi))
            if wrong_bank_key in eng.flipped_row_keys:
                break
        # The target row *index* really did flip — in the wrong bank. Nothing may
        # have been written at the task's own (bank 0) target address.
        self.assertIn(wrong_bank_key, eng.flipped_row_keys)
        self.assertNotIn(ct.target_row_key, eng.flipped_row_keys)
        self.assertNotIn(ct.target_addr, eng.flips)
        self.assertFalse(success_for(ct, eng, _unexpected_read))

    def test_any_flip_predicate(self) -> None:
        ct = compile_task({"family": "any_flip"}, seed=4)
        eng = self._engine_for(ct)
        self.assertFalse(success_for(ct, eng, _unexpected_read))
        eng.flips[999 * eng.row_bytes] = 2  # any cell anywhere
        self.assertTrue(success_for(ct, eng, _unexpected_read))

    def test_target_cell_predicate_reads_the_actual_post_disturbance_bit(self) -> None:
        # @spec:rl-reward @spec:invariant-trusted-reward
        ct = compile_task({"family": "target_cell", "bit": 0}, seed=4)
        eng = self._engine_for(ct)
        eng.flips[ct.target_addr] = 3  # wrong bit
        self.assertFalse(success_for(ct, eng, _unexpected_read))

        eng.flips[ct.target_addr] = 0
        # A zero stored byte reads as 0x01 after the flip and satisfies the default
        # target value. An 0xff stored byte reads as 0xfe, so the same committed
        # flip does not satisfy it; the flip map alone cannot decide success.
        self.assertTrue(success_for(ct, eng, _stored_byte_reader(eng, 0x00)))
        self.assertFalse(success_for(ct, eng, _stored_byte_reader(eng, 0xFF)))

    def test_pattern_predicate_reads_the_actual_post_disturbance_byte(self) -> None:
        # @spec:rl-reward @spec:invariant-trusted-reward
        ct = compile_task({"family": "pattern_target", "bit": 0}, seed=4)
        eng = self._engine_for(ct)
        self.assertFalse(success_for(ct, eng, _unexpected_read))
        eng.flips[ct.target_addr] = ct.target_bit
        self.assertTrue(success_for(ct, eng, _stored_byte_reader(eng, 0x00)))
        self.assertFalse(success_for(ct, eng, _stored_byte_reader(eng, 0xFF)))


class TaskConfigFileTests(unittest.TestCase):
    def test_all_configs_validate_and_compile_and_cover_families(self) -> None:
        seen = set()
        configs = sorted(TASK_DIR.glob("*.yaml"))
        self.assertTrue(configs)
        for path in configs:
            doc = yaml.safe_load(path.read_text())
            jsonschema.validate(doc, SCHEMA)
            spec = TaskSpec.from_config(doc)
            seen.add(spec.family)
            if FAMILIES[spec.family].secret_mapping:
                # Without the worker decoder these would compile to a bank-0 victim
                # key and an empty candidate window — a CompiledTask that is wrong for
                # any real episode — so they must fail closed. Their real compile runs
                # against the worker in TaskEnvIntegrationTests / verify_phase13.
                with self.assertRaises(TaskConfigError):
                    spec.compile(seed=1, geometry=geo())
                continue
            ct = spec.compile(seed=1, geometry=geo())
            self.assertEqual(ct.family, spec.family)
            self.assertIn("acts", ct.budgets)
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

    def test_trusted_success_latches_across_overlay_retraction(self) -> None:
        # @spec:rl-reward @spec:rl-episode-termination
        env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
        try:
            env.reset(seed=21)
            assert env._compiled is not None and env.disturbance is not None
            key = env._compiled.target_row_key

            env.disturbance.flipped_row_keys.add(key)
            self.assertTrue(env._latch_success())
            env.disturbance.flipped_row_keys.remove(key)
            self.assertFalse(env._trusted_success())

            # The raw predicate reflects the retracted overlay, but decision B
            # makes the episode reward monotone once trusted success was observed.
            fin = env.step(Phase2Action(tool="episode.finish", args={}))
            self.assertEqual(fin.reward, 1.0)
            self.assertTrue(fin.done)
        finally:
            env.close()

    def test_byte_predicates_read_worker_storage_and_apply_disturbance(self) -> None:
        # @spec:rl-reward @spec:invariant-trusted-reward
        # The trusted read is side-effect-free: checking reward must not tick the
        # worker, increment counters, or create activations outside policy budgets.
        for family in ("target_cell", "pattern_target"):
            with self.subTest(family=family):
                env = RowHammerTaskEnv(task={"family": family, "bit": 0})
                try:
                    env.reset(seed=4)
                    assert env._compiled is not None and env.disturbance is not None
                    addr = env._compiled.target_addr

                    write_ff = env.step(Phase2Action(tool="dram.write", args={
                        "addr": {"kind": "logical", "addr": addr},
                        "data_b64": base64.b64encode(b"\xff").decode(),
                    }))
                    self.assertIsNone(write_ff.error)
                    env.disturbance.flips[addr] = 0
                    cycle_before = env.state.cycle
                    counters_before = dict(env._public_counters)
                    self.assertFalse(env._trusted_success())  # 0xff ^ 0x01 == 0xfe
                    self.assertEqual(env.state.cycle, cycle_before)
                    self.assertEqual(env._public_counters, counters_before)

                    write_zero = env.step(Phase2Action(tool="dram.write", args={
                        "addr": {"kind": "logical", "addr": addr},
                        "data_b64": base64.b64encode(b"\x00").decode(),
                    }))
                    self.assertIsNone(write_zero.error)
                    env.disturbance.flips[addr] = 0
                    self.assertTrue(env._trusted_success())  # 0x00 ^ 0x01 == 0x01
                finally:
                    env.close()

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
