"""Tier 2a discovery family — bounded addressable candidate window (P23/P24).

``bounded_sweep`` registers ``N`` RD-addressable candidate handles at a deliberate
mix of same-bank/different-bank, adjacent/far offsets, only the two immediate
same-bank neighbours of which are true aggressors of the (handle-hidden) victim.
Since P24 the family runs under a **per-episode secret row->bank address mapper**,
so bank membership is not computable from the linear address and the candidate
window is built against the true mapping via the worker ``DECODE`` op. These tests
drive the real worker-gated ``RowHammerTaskEnv`` and assert the "Done when"
contract:

* each band (``easy``/``medium``/``hard``) instantiates from its shipped config,
  compiles, and runs;
* a fixed-seed fixture confirms the intended true-aggressor fraction (exactly the
  two immediate neighbours) **and** the intended same/different-bank split, decoded
  against the real (secret) mapper via ``DECODE``;
* hammering the true aggressors flips the victim while hammering only the decoys
  never does — a candidate's identity as "the real one" is derivable only from the
  trusted final flip, never from the handle id, its list position, or any disclosed
  field.
"""

from __future__ import annotations

import base64
import pathlib
import unittest
from collections import Counter

import yaml

from rowhammer_env import Phase2Action, RowHammerTaskEnv
from rowhammer_env.tasks.compiler import BAND_CANDIDATES, FAR_ROW_MARGIN
from rowhammer_env.tools.addressing import COORD_KEYS

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
CONFIGS = {band: ROOT / f"configs/tasks/bounded_sweep_{band}.yaml" for band in ("easy", "medium", "hard")}


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class CandidateWindowTests(unittest.TestCase):
    """Compile ``bounded_sweep`` through the real env and verify the window."""

    def _reset(self, band: str, seed: int = 7) -> RowHammerTaskEnv:
        env = RowHammerTaskEnv(task={"family": "bounded_sweep", "difficulty": band, "id": f"bs_{band}"})
        obs = env.reset(seed=seed)
        self.assertIsNone(obs.error, band)
        return env

    def _roles_decoded(self, env: RowHammerTaskEnv):
        """(role, decoded_coords) for every candidate, decoded via the secret mapper."""
        ct = env._compiled
        out = []
        for c in ct.candidates:
            out.append((c.role, env._decode(ct.target_addr + c.offset)))
        return out

    def test_band_candidate_counts(self) -> None:
        for band, n in BAND_CANDIDATES.items():
            env = self._reset(band)
            try:
                self.assertEqual(len(env._compiled.candidates), n, band)
                self.assertEqual(len(env.reset(seed=7).metadata["candidates"]), n, band)
            finally:
                env.close()

    def test_exactly_two_true_aggressors_are_immediate_neighbours(self) -> None:
        for band in BAND_CANDIDATES:
            env = self._reset(band)
            try:
                ct = env._compiled
                vbank = (ct.target_bankgroup, ct.target_bank)
                rows = set()
                for role, d in self._roles_decoded(env):
                    if role != "aggressor":
                        continue
                    self.assertEqual((d["bankgroup"], d["bank"]), vbank, (band, "aggressor same bank"))
                    self.assertEqual(abs(d["row"] - ct.target_row), 1, band)
                    rows.add(d["row"])
                self.assertEqual(rows, {ct.target_row - 1, ct.target_row + 1}, band)
            finally:
                env.close()

    def test_bank_split_decoded_against_secret_mapper(self) -> None:
        for band in BAND_CANDIDATES:
            env = self._reset(band, seed=13)
            try:
                ct = env._compiled
                vbank = (ct.target_bankgroup, ct.target_bank)
                for role, d in self._roles_decoded(env):
                    same = (d["bankgroup"], d["bank"]) == vbank
                    if role == "different_bank":
                        self.assertFalse(same, (band, d))
                    else:  # aggressor | same_bank_far
                        self.assertTrue(same, (band, role, d))
                    if role == "same_bank_far":
                        self.assertGreater(abs(d["row"] - ct.target_row), FAR_ROW_MARGIN, (band, d))
            finally:
                env.close()

    def test_decoy_split_is_roughly_balanced(self) -> None:
        env = self._reset("hard", seed=1)
        try:
            roles = Counter(c.role for c in env._compiled.candidates)
            self.assertEqual(roles["aggressor"], 2)
            self.assertEqual(roles["same_bank_far"] + roles["different_bank"], BAND_CANDIDATES["hard"] - 2)
            self.assertLessEqual(abs(roles["same_bank_far"] - roles["different_bank"]), 1)
        finally:
            env.close()

    def test_candidate_window_is_deterministic_per_seed(self) -> None:
        a = self._reset("medium", seed=7)
        b = self._reset("medium", seed=7)
        try:
            self.assertEqual(a._compiled.candidates, b._compiled.candidates)
        finally:
            a.close()
            b.close()

    def test_aggressor_position_is_not_fixed_across_seeds(self) -> None:
        positions = set()
        for seed in range(8):
            env = self._reset("medium", seed=seed)
            try:
                positions.update(i for i, c in enumerate(env._compiled.candidates) if c.is_aggressor)
            finally:
                env.close()
        self.assertGreater(len(positions), 2, positions)


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class BoundedSweepIntegrationTests(unittest.TestCase):
    def _task(self, band: str) -> dict:
        return yaml.safe_load(CONFIGS[band].read_text())

    def _handle(self, handle_id: str) -> dict:
        return {"kind": "handle", "id": handle_id}

    def test_each_band_instantiates_compiles_and_runs(self) -> None:
        for band, n in BAND_CANDIDATES.items():
            env = RowHammerTaskEnv(task=self._task(band))
            try:
                obs = env.reset(seed=7)
                self.assertIsNone(obs.error, band)
                self.assertEqual(env._compiled.family, "bounded_sweep", band)
                self.assertEqual(len(obs.metadata["candidates"]), n, band)
                step = env.step(Phase2Action(tool="dram.issue", args={
                    "commands": [{"op": "RD", "addr": obs.metadata["candidates"][0]}]}))
                self.assertIsNone(step.error, band)
            finally:
                env.close()

    def test_hammering_the_true_aggressors_flips_the_victim(self) -> None:
        env = RowHammerTaskEnv(task=self._task("easy"))
        try:
            obs = env.reset(seed=7)
            cands = env._compiled.candidates
            meta = obs.metadata["candidates"]
            aggr = [meta[i]["id"] for i, c in enumerate(cands) if c.is_aggressor]
            thr = env.disturbance.known_threshold
            last = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "HAMMER", "rows": [self._handle(aggr[0]), self._handle(aggr[1])],
                              "pairs": thr}]}))
            self.assertEqual(last.reward, 1.0)
            self.assertTrue(last.done)
        finally:
            env.close()

    def test_dram_read_shows_the_flip_at_the_disclosed_victim_handle(self) -> None:
        # @spec:tool-dram-read — a committed flip must be readable at the address the
        # task disclosed. The victim's anchor is its own column-0 address under the
        # per-episode secret mapper (resolved through the worker ``ENCODE`` op), so
        # resolving the target handle and reading it returns the flipped byte.
        env = RowHammerTaskEnv(task=self._task("easy"))
        try:
            obs = env.reset(seed=7)
            cands = env._compiled.candidates
            meta = obs.metadata["candidates"]
            aggr = [self._handle(meta[i]["id"]) for i, c in enumerate(cands) if c.is_aggressor]
            target = obs.metadata["objective"]["target"]
            self.assertEqual(target["kind"], "handle")
            step = env.step(Phase2Action(tool="dram.issue", args={"commands": [
                {"op": "HAMMER", "rows": aggr, "pairs": env.disturbance.known_threshold},
                {"op": "RD", "addr": target},
            ]}))
            self.assertIsNone(step.error)
            self.assertEqual(step.reward, 1.0)
            self.assertNotEqual(base64.b64decode(step.data_b64)[0], 0)
        finally:
            env.close()

    def test_hammering_only_decoys_does_not_flip(self) -> None:
        # The "real one" is derivable only from the trusted flip: hammering the
        # non-aggressor candidates (same-bank-far + different-bank) never succeeds.
        env = RowHammerTaskEnv(task=self._task("medium"))
        try:
            obs = env.reset(seed=7)
            cands = env._compiled.candidates
            meta = obs.metadata["candidates"]
            decoys = [self._handle(meta[i]["id"]) for i, c in enumerate(cands) if not c.is_aggressor]
            thr = env.disturbance.known_threshold
            last = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "HAMMER", "rows": decoys, "pairs": thr}]}))
            self.assertEqual(last.reward, 0.0)
            fin = env.step(Phase2Action(tool="episode.finish", args={}))
            self.assertEqual(fin.reward, 0.0)
        finally:
            env.close()

    def test_no_candidate_identity_leaks_in_disclosed_surface(self) -> None:
        env = RowHammerTaskEnv(task=self._task("hard"))
        try:
            obs = env.reset(seed=7)
            meta = obs.metadata
            for cand in meta["candidates"]:
                self.assertEqual(set(cand), {"kind", "id"})
                self.assertEqual(cand["kind"], "handle")
                self.assertTrue(str(cand["id"]).startswith("h_"))
            blob = repr(meta)
            for c in env._compiled.candidates:
                self.assertNotIn(str(env._compiled.target_addr + c.offset), blob)
            for token in ("aggressor", "same_bank_far", "different_bank"):
                self.assertNotIn(token, blob)
            for coord in COORD_KEYS:
                self.assertNotIn(f"'{coord}'", blob)
        finally:
            env.close()

    def test_probe_timing_separates_banks_through_the_real_env(self) -> None:
        # The disclosed signal is load-bearing: alternating two same-bank candidates
        # forces new ACTs; pairing one with a different-bank candidate does not
        # (§0.2) — the discriminator is ``acts_delta``, all keyed on opaque handles.
        env = RowHammerTaskEnv(task=self._task("hard"))
        try:
            obs = env.reset(seed=7)
            cands = env._compiled.candidates
            meta = obs.metadata["candidates"]

            def pick(role: str) -> dict:
                return self._handle(meta[next(i for i, c in enumerate(cands) if c.role == role)]["id"])

            aggr = pick("aggressor")
            same_far = pick("same_bank_far")
            diff = pick("different_bank")

            def probe(a: dict, b: dict) -> dict:
                env.step(Phase2Action(tool="dram.issue", args={"commands": [
                    {"op": "RD", "addr": a}, {"op": "RD", "addr": b}]}))
                step = env.step(Phase2Action(tool="dram.issue", args={
                    "commands": [{"op": "HAMMER", "rows": [a, b], "pairs": 8}]}))
                return step.feedback["timing_digest"]

            same_digest = probe(aggr, same_far)
            diff_digest = probe(aggr, diff)
            self.assertGreater(same_digest["acts_delta"], 0)
            self.assertEqual(diff_digest["acts_delta"], 0)
            self.assertGreater(same_digest["cycles_delta"], diff_digest["cycles_delta"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
