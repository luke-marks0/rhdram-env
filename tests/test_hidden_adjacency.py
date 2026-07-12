"""Tier 2b discovery family — known numeric address, secret adjacency (P25).

``hidden_adjacency`` is the real-attacker-knowledge model (``TIER2_DISCOVERY_PLAN``
§4.1): the victim's own **numeric (logical) address** and a set of **numeric**
candidate addresses are disclosed, but the address->bank mapping is a per-episode
secret (P24), so which candidates are same-bank + physically adjacent must be
reverse-engineered from the bank-conflict timing channel (§0.2) — not computed from
the address. The two true aggressors (victim's immediate physical neighbours, same
bank, rows ± 1) are camouflaged by different-bank decoys at the *same* adjacent
rows, so no disclosed field (numeric address, timing digest, error) reveals which
candidate is the real one — only the timing channel and the trusted final flip can.

These tests drive the real worker-gated ``RowHammerTaskEnv`` (a P24 worker rebuild
is required — the C++ ``DECODE`` op and the ``RoBaRaCoChRowXOR`` mapper) and assert
the P25 "Done when" contract:

* the family instantiates, compiles, and runs from its shipped band configs, with
  numeric (not handle) candidates;
* a fixed-seed fixture confirms the true-aggressor fraction (exactly the two
  immediate neighbours, same bank) and the same/different-bank split, decoded
  against the real secret mapper via ``DECODE``;
* a control that computes ``victim ± row_stride`` and hammers it **fails** (the
  row->bank XOR sends those addresses to a different bank), while hammering the true
  aggressors flips the decoded victim;
* the timing channel is load-bearing (a candidate probed against the disclosed
  numeric victim classifies same/different bank via ``acts_delta``);
* no disclosed field leaks a candidate's bank/role or the mapper.
"""

from __future__ import annotations

import pathlib
import unittest
from collections import Counter

import yaml

from rowhammer_env import Phase2Action, RowHammerTaskEnv
from rowhammer_env.phase5_env import ALLOWED_TOOLS
from rowhammer_env.tasks.compiler import BAND_CANDIDATES, FAMILIES, FAR_ROW_MARGIN, TaskSpec
from rowhammer_env.tools.addressing import COORD_KEYS

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
CONFIGS = {band: ROOT / f"configs/tasks/hidden_adjacency_{band}.yaml" for band in ("easy", "medium", "hard")}


class HiddenAdjacencyRegistrationTests(unittest.TestCase):
    """Family/predicate/disclosure wiring — no worker needed."""

    def test_family_is_registered_as_a_numeric_secret_mapping_discovery_family(self) -> None:
        fam = FAMILIES["hidden_adjacency"]
        self.assertTrue(fam.secret_mapping)
        self.assertEqual(fam.disclosure.mapping, "logical_only")
        self.assertEqual(fam.disclosure.victim, "logical_addr")
        self.assertEqual(fam.disclosure.adjacency, "candidate_set")
        self.assertEqual(fam.disclosure.feedback, "full_trace")

    def test_logical_addr_victim_exposes_the_address_but_not_coordinates(self) -> None:
        disc = FAMILIES["hidden_adjacency"].disclosure
        self.assertTrue(disc.expose_victim_address())
        self.assertFalse(disc.expose_victim())  # no physical coords / threshold
        self.assertFalse(disc.expose_coords())
        # logical_addr is not a handle level: no opaque handles are minted.
        self.assertFalse(disc.handles_allowed())
        self.assertEqual(disc.allowed_forms(), {"logical"})

    def test_full_config_without_family_name_derives_hidden_adjacency(self) -> None:
        cfg = {
            "objective": {"type": "target_row_flip"},
            "disclosure": {"mapping": "logical_only", "adjacency": "candidate_set", "victim": "logical_addr"},
        }
        self.assertEqual(TaskSpec.from_config(cfg).family, "hidden_adjacency")

    def test_predicate_reads_the_decoded_victim_key(self) -> None:
        from rowhammer_env.rewards.predicates import PREDICATES, _target_bankrow_flip

        self.assertIs(PREDICATES["hidden_adjacency"], _target_bankrow_flip)


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class NumericCandidateWindowTests(unittest.TestCase):
    """Compile ``hidden_adjacency`` through the real env and verify the window."""

    def _reset(self, band: str, seed: int = 7) -> RowHammerTaskEnv:
        env = RowHammerTaskEnv(task={"family": "hidden_adjacency", "difficulty": band, "id": f"ha_{band}"})
        obs = env.reset(seed=seed)
        self.assertIsNone(obs.error, band)
        return env

    def _roles_decoded(self, env: RowHammerTaskEnv):
        ct = env._compiled
        return [(c.role, env._decode(ct.target_addr + c.offset)) for c in ct.candidates]

    def test_secret_mapper_is_active_and_logical_only(self) -> None:
        env = self._reset("easy")
        try:
            self.assertEqual(env._active_mapper_impl, "RoBaRaCoChRowXOR")
            self.assertIn("xor_offset", env._active_mapper_params)
            self.assertEqual(env.disclosure.mapping, "logical_only")
        finally:
            env.close()

    def test_band_candidate_counts(self) -> None:
        for band, n in BAND_CANDIDATES.items():
            env = self._reset(band)
            try:
                self.assertEqual(len(env._compiled.candidates), n, band)
                self.assertEqual(len(env.reset(seed=7).metadata["candidates"]), n, band)
            finally:
                env.close()

    def test_candidates_are_numeric_logical_addresses_not_handles(self) -> None:
        env = self._reset("medium")
        try:
            obs = env.reset(seed=7)
            ct = env._compiled
            target = obs.metadata["objective"]["target"]
            self.assertEqual(target, {"kind": "logical", "addr": ct.target_addr})
            self.assertEqual(obs.metadata["target"], {"kind": "logical", "addr": ct.target_addr})
            for i, cand in enumerate(obs.metadata["candidates"]):
                self.assertEqual(set(cand), {"kind", "addr"})
                self.assertEqual(cand["kind"], "logical")
                # Numeric address matches the compiled candidate offset in order.
                self.assertEqual(cand["addr"], ct.target_addr + ct.candidates[i].offset)
            # No opaque handles exist for this family at all.
            self.assertEqual(env._candidate_handles, [])
            self.assertIsNone(env._target_handle)
        finally:
            env.close()

    def test_exactly_two_true_aggressors_are_immediate_same_bank_neighbours(self) -> None:
        for band in BAND_CANDIDATES:
            env = self._reset(band, seed=13)
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
            env = self._reset(band, seed=21)
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

    def test_public_row_does_not_isolate_the_aggressors(self) -> None:
        # The one thing a policy *can* compute from a numeric address is its row (the
        # stride is public geometry). So every adjacent row that hosts an aggressor
        # must also host a different-bank decoy — otherwise filtering candidates by
        # "row == victim_row ± 1" would reveal the aggressors without any timing.
        for band in BAND_CANDIDATES:
            env = self._reset(band, seed=5)
            try:
                dec = [(role, d["row"]) for role, d in self._roles_decoded(env)]
                agg_rows = {row for role, row in dec if role == "aggressor"}
                diff_rows = {row for role, row in dec if role == "different_bank"}
                self.assertTrue(agg_rows <= diff_rows, (band, agg_rows, diff_rows))
            finally:
                env.close()

    def test_window_is_deterministic_and_aggressor_position_varies(self) -> None:
        a = self._reset("medium", seed=7)
        b = self._reset("medium", seed=7)
        try:
            self.assertEqual(a._compiled.candidates, b._compiled.candidates)
        finally:
            a.close()
            b.close()
        positions = set()
        for seed in range(8):
            env = self._reset("medium", seed=seed)
            try:
                positions.update(i for i, c in enumerate(env._compiled.candidates) if c.is_aggressor)
            finally:
                env.close()
        self.assertGreater(len(positions), 2, positions)


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class HiddenAdjacencyIntegrationTests(unittest.TestCase):
    def _task(self, band: str) -> dict:
        return yaml.safe_load(CONFIGS[band].read_text())

    def _addr(self, meta_cand: dict) -> dict:
        return {"kind": "logical", "addr": meta_cand["addr"]}

    def test_each_band_instantiates_compiles_and_runs(self) -> None:
        for band, n in BAND_CANDIDATES.items():
            env = RowHammerTaskEnv(task=self._task(band))
            try:
                obs = env.reset(seed=7)
                self.assertIsNone(obs.error, band)
                self.assertEqual(env._compiled.family, "hidden_adjacency", band)
                self.assertEqual(len(obs.metadata["candidates"]), n, band)
                step = env.step(Phase2Action(tool="dram.issue", args={
                    "commands": [{"op": "RD", "addr": obs.metadata["candidates"][0]}]}))
                self.assertIsNone(step.error, band)
            finally:
                env.close()

    def test_hammering_the_true_aggressors_flips_the_decoded_victim(self) -> None:
        env = RowHammerTaskEnv(task=self._task("easy"),
                               budgets={"tool_calls": 60, "acts": 120_000, "cycles": 240_000_000})
        try:
            obs = env.reset(seed=7)
            ct = env._compiled
            meta = obs.metadata["candidates"]
            aggr = [self._addr(meta[i]) for i, c in enumerate(ct.candidates) if c.is_aggressor]
            thr = env.disturbance.known_threshold
            last = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "HAMMER", "rows": aggr, "pairs": thr}]}))
            self.assertEqual(last.reward, 1.0)
            self.assertTrue(last.done)
            key = (0, 0, ct.target_bankgroup, ct.target_bank, ct.target_row)
            self.assertIn(key, env.disturbance.flipped_row_keys)
        finally:
            env.close()

    def test_arithmetic_control_victim_plus_minus_stride_fails(self) -> None:
        # The P25 done-when: a real-attacker control that computes ``victim ± row_stride``
        # from the disclosed numeric address and hammers it fails — the secret row->bank
        # XOR sends those addresses to a different bank, so no exposure reaches the victim.
        env = RowHammerTaskEnv(task=self._task("medium"),
                               budgets={"tool_calls": 60, "acts": 120_000, "cycles": 240_000_000})
        try:
            obs = env.reset(seed=7)
            victim = obs.metadata["objective"]["target"]["addr"]
            rb = env._compiled.row_bytes
            thr = env.disturbance.known_threshold
            last = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "HAMMER", "rows": [victim - rb, victim + rb], "pairs": thr}]}))
            self.assertEqual(last.reward, 0.0)
            fin = env.step(Phase2Action(tool="episode.finish", args={}))
            self.assertEqual(fin.reward, 0.0)
        finally:
            env.close()

    def test_hammering_only_decoys_does_not_flip(self) -> None:
        env = RowHammerTaskEnv(task=self._task("medium"),
                               budgets={"tool_calls": 60, "acts": 200_000, "cycles": 240_000_000})
        try:
            obs = env.reset(seed=7)
            ct = env._compiled
            meta = obs.metadata["candidates"]
            decoys = [self._addr(meta[i]) for i, c in enumerate(ct.candidates) if not c.is_aggressor]
            thr = env.disturbance.known_threshold
            last = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "HAMMER", "rows": decoys, "pairs": thr}]}))
            self.assertEqual(last.reward, 0.0)
            fin = env.step(Phase2Action(tool="episode.finish", args={}))
            self.assertEqual(fin.reward, 0.0)
        finally:
            env.close()

    def test_timing_channel_classifies_candidate_against_the_numeric_victim(self) -> None:
        # The disclosed signal is load-bearing: probing a candidate against the
        # disclosed numeric victim address forces a new ACT iff they share a bank
        # (§0.2) — the discriminator is ``acts_delta``, never the RD's own row_hit.
        env = RowHammerTaskEnv(task=self._task("hard"))
        try:
            obs = env.reset(seed=7)
            ct = env._compiled
            victim = obs.metadata["objective"]["target"]
            meta = obs.metadata["candidates"]

            def pick(role: str) -> dict:
                return self._addr(meta[next(i for i, c in enumerate(ct.candidates) if c.role == role)])

            aggr = pick("aggressor")
            diff = pick("different_bank")

            def probe(a: dict, b: dict) -> dict:
                env.step(Phase2Action(tool="dram.issue", args={"commands": [
                    {"op": "RD", "addr": a}, {"op": "RD", "addr": b}]}))
                step = env.step(Phase2Action(tool="dram.issue", args={
                    "commands": [{"op": "HAMMER", "rows": [a, b], "pairs": 8}]}))
                return step.feedback["timing_digest"]

            same = probe(aggr, victim)
            other = probe(diff, victim)
            self.assertGreater(same["acts_delta"], 0)
            self.assertEqual(other["acts_delta"], 0)
            self.assertGreater(same["cycles_delta"], other["cycles_delta"])
        finally:
            env.close()

    def test_no_candidate_identity_or_mapper_leaks_in_disclosed_surface(self) -> None:
        env = RowHammerTaskEnv(task=self._task("hard"))
        try:
            obs = env.reset(seed=7)
            offset = str(env._active_mapper_params["xor_offset"])
            info = env.step(Phase2Action(tool="dram.info", args={}))
            issue = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "RD", "addr": obs.metadata["candidates"][0]}]}))
            surfaces = [repr(obs.metadata), repr(info.metadata), repr(info.info),
                        repr(issue.feedback), repr(issue.metadata)]
            for blob in surfaces:
                # No physical coordinates, no role tags, no mapper identity anywhere.
                for coord in COORD_KEYS:
                    self.assertNotIn(f"'{coord}'", blob)
                for token in ("aggressor", "same_bank_far", "different_bank"):
                    self.assertNotIn(token, blob)
                for banned in ("RoBaRaCoChRowXOR", "xor_offset", "RoBaRaCoCh", "MOP4CLXOR", "addr_mapper"):
                    self.assertNotIn(banned, blob)
                self.assertNotIn(f"'xor_offset': {offset}", blob)
            # The victim's decoded bank must not be recoverable from any disclosed
            # field — only the row (public geometry) is derivable from the address.
            self.assertNotIn(f"'bank': {env._compiled.target_bank}", repr(obs.metadata))
        finally:
            env.close()

    def test_decode_is_not_a_policy_tool(self) -> None:
        env = RowHammerTaskEnv(task=self._task("easy"))
        try:
            env.reset(seed=7)
            self.assertNotIn("DECODE", ALLOWED_TOOLS)
            step = env.step(Phase2Action(tool="DECODE", args={"addr": 0}))
            self.assertEqual(step.error["code"], "UNSUPPORTED_TOOL")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
