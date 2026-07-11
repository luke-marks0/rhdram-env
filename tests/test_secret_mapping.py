"""Realistic secret address mapping — the discovery linchpin (P24).

For discovery families the address->bank function is a per-episode secret, so a
policy given a numeric victim address cannot compute which candidates are same-bank
neighbours — adjacency must be reverse-engineered by the bank-conflict timing
channel (DRAMA). Ramulator's stock mappers do not provide this (RoBaRaCoCh keeps
the bank a pure low-bit slice; MOP4CLXOR XORs *column* bits into the bank, so
``addr + row_stride`` stays same-bank under both), so the environment ships an
authored, source-cited ``RoBaRaCoChRowXOR`` mapper that XORs *row* bits into the
bank index — real controller behaviour (Pessl et al., DRAMA), not fabricated
physics. See ``docs/adr-0004-secret-address-mapping.md``.

These tests drive the real worker (a rebuild is required — the C++ ``DECODE`` op
and the new mapper). They prove: the mapper genuinely scatters adjacency across
banks while preserving the disclosed geometry; the worker ``DECODE`` op reports the
true coordinates and is never a policy tool; per-episode selection is deterministic
and secret; a numeric ``victim ± row_stride`` control fails; the disturbance still
flips the decoded aggressor->victim pair; and no disclosed field leaks the mapper.
"""

from __future__ import annotations

import pathlib
import unittest

from rowhammer_env import Phase2Action, RowHammerTaskEnv
from rowhammer_env.mappers import (
    DEFAULT_MAPPER,
    SECRET_MAPPERS,
    is_python_projectable,
    select_secret_mapper,
    worker_config_for_mapper,
)
from rowhammer_env.phase5_env import ALLOWED_TOOLS
from rowhammer_env.worker_protocol import WorkerClient, WorkerRequest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
BASE_CONFIG = ROOT / "build/phase2/p2_external_ddr4.yaml"
ROW_BYTES = 131072  # the DDR4_8Gb_x8 row stride, identical across all these mappers


class MapperSelectionUnitTests(unittest.TestCase):
    def test_secret_set_is_all_row_scattering(self) -> None:
        # Every secret mapper is the authored row->bank XOR (the only one that makes
        # adjacency non-computable); the seedable xor_offset is the per-episode secret.
        self.assertTrue(SECRET_MAPPERS)
        for impl, params in SECRET_MAPPERS:
            self.assertEqual(impl, "RoBaRaCoChRowXOR")
            self.assertIn("xor_offset", params)

    def test_selection_is_deterministic_per_task_and_seed(self) -> None:
        a = select_secret_mapper("ddr4_x", 7)
        self.assertEqual(a, select_secret_mapper("ddr4_x", 7))
        # Varying either the task id or the seed can change the choice.
        variants = {(impl, tuple(sorted(p.items()))) for impl, p in
                    (select_secret_mapper("ddr4_x", s) for s in range(20))}
        self.assertGreater(len(variants), 1)

    def test_only_default_mapper_is_python_projectable(self) -> None:
        self.assertTrue(is_python_projectable(DEFAULT_MAPPER))
        self.assertFalse(is_python_projectable("RoBaRaCoChRowXOR"))

    def test_default_mapper_reuses_base_config_verbatim(self) -> None:
        self.assertEqual(worker_config_for_mapper(BASE_CONFIG, DEFAULT_MAPPER, {}), BASE_CONFIG)


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class DecodeAndScatterTests(unittest.TestCase):
    def _client(self, impl: str, params: dict) -> WorkerClient:
        cfg = worker_config_for_mapper(BASE_CONFIG, impl, params)
        return WorkerClient(WORKER, cfg)

    def _decode(self, cli: WorkerClient, linear: int) -> dict:
        resp = cli.call(WorkerRequest("DECODE", "d", (str(linear),)))
        self.assertTrue(resp.get("ok"), resp)
        return resp["addr_vec"]

    def test_decode_reports_full_coordinates(self) -> None:
        cli = self._client(DEFAULT_MAPPER, {})
        try:
            d = self._decode(cli, 5 * ROW_BYTES)
            self.assertEqual(set(d), {"channel", "rank", "bankgroup", "bank", "row", "column"})
            # Under RoBaRaCoCh, row is the MSB field: linear // row_bytes == row.
            self.assertEqual(d["row"], 5)
            self.assertEqual((d["bankgroup"], d["bank"]), (0, 0))
        finally:
            cli.close()

    def test_row_xor_mapper_scatters_adjacency_across_banks(self) -> None:
        # The load-bearing property: `addr` and `addr + row_stride` land in DIFFERENT
        # banks for (documented) 100% of addresses, while the row is preserved and
        # the stride is unchanged — so the disclosed geometry (P21) stays honest.
        for _impl, params in SECRET_MAPPERS:
            cli = self._client("RoBaRaCoChRowXOR", params)
            try:
                diff = 0
                total = 200
                for i in range(total):
                    base = (i * 211 + 1) * ROW_BYTES  # spread across rows
                    a = self._decode(cli, base)
                    b = self._decode(cli, base + ROW_BYTES)
                    self.assertEqual(b["row"] - a["row"], 1)  # row preserved / stride intact
                    if (a["bankgroup"], a["bank"]) != (b["bankgroup"], b["bank"]):
                        diff += 1
                self.assertEqual(diff, total, params)  # every adjacent row scatters
            finally:
                cli.close()

    def test_stock_mappers_do_not_scatter_adjacency(self) -> None:
        # Documents *why* an authored mapper is needed: neither RoBaRaCoCh nor
        # MOP4CLXOR moves the adjacent row into a different bank.
        for impl in ("RoBaRaCoCh", "MOP4CLXOR"):
            cli = self._client(impl, {})
            try:
                diff = 0
                for i in range(100):
                    base = (i * 307 + 1) * ROW_BYTES
                    a = self._decode(cli, base)
                    b = self._decode(cli, base + ROW_BYTES)
                    if (a["bankgroup"], a["bank"]) != (b["bankgroup"], b["bank"]):
                        diff += 1
                self.assertEqual(diff, 0, impl)
            finally:
                cli.close()


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class SecretMappingEnvTests(unittest.TestCase):
    def _bounded_sweep(self, seed: int = 7, band: str = "easy", **kw) -> RowHammerTaskEnv:
        budgets = kw.pop("budgets", {"tool_calls": 60, "acts": 80_000, "cycles": 160_000_000})
        env = RowHammerTaskEnv(task={"family": "bounded_sweep", "difficulty": band, "id": "ddr4_bs"},
                               budgets=budgets)
        obs = env.reset(seed=seed)
        self.assertIsNone(obs.error)
        return env

    def test_discovery_family_uses_a_secret_mapper(self) -> None:
        env = self._bounded_sweep()
        try:
            self.assertEqual(env._active_mapper_impl, "RoBaRaCoChRowXOR")
            self.assertIn("xor_offset", env._active_mapper_params)
        finally:
            env.close()

    def test_non_discovery_family_stays_public_robaracoch(self) -> None:
        env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
        try:
            env.reset(seed=7)
            self.assertEqual(env._active_mapper_impl, DEFAULT_MAPPER)
            self.assertFalse(env._active_mapper_params)
        finally:
            env.close()

    def test_decode_is_not_a_policy_tool(self) -> None:
        env = self._bounded_sweep()
        try:
            self.assertNotIn("DECODE", ALLOWED_TOOLS)
            self.assertNotIn("dram.decode", ALLOWED_TOOLS)
            # A policy DECODE attempt is an unsupported tool, not a decode.
            step = env.step(Phase2Action(tool="DECODE", args={"addr": 0}))
            self.assertEqual(step.error["code"], "UNSUPPORTED_TOOL")
        finally:
            env.close()

    def test_disturbance_flips_the_decoded_aggressor_victim_pair(self) -> None:
        env = self._bounded_sweep()
        try:
            ct = env._compiled
            meta = env.reset(seed=7).metadata["candidates"]  # order matches candidates
            cands = env._compiled.candidates
            aggr = [meta[i]["id"] for i, c in enumerate(cands) if c.is_aggressor]
            thr = env.disturbance.known_threshold
            last = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "HAMMER", "rows": [{"kind": "handle", "id": aggr[0]},
                                                       {"kind": "handle", "id": aggr[1]}], "pairs": thr}]}))
            self.assertEqual(last.reward, 1.0)
            # The trusted flip is keyed on the *decoded* victim (bankgroup, bank, row).
            key = (0, 0, ct.target_bankgroup, ct.target_bank, ct.target_row)
            self.assertIn(key, env.disturbance.flipped_row_keys)
        finally:
            env.close()

    def test_arithmetic_control_fails_under_secret_mapping(self) -> None:
        # A real-attacker control that computes `victim ± row_stride` and hammers it
        # fails: the secret row->bank XOR sends those addresses to a different bank,
        # so no exposure reaches the victim (the P25 acceptance criterion).
        env = self._bounded_sweep(budgets={"tool_calls": 60, "acts": 120_000, "cycles": 240_000_000})
        try:
            ct = env._compiled
            rb = ct.row_bytes
            thr = env.disturbance.known_threshold
            last = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "HAMMER", "rows": [ct.target_addr - rb, ct.target_addr + rb],
                              "pairs": thr}]}))
            self.assertEqual(last.reward, 0.0)
            fin = env.step(Phase2Action(tool="episode.finish", args={}))
            self.assertEqual(fin.reward, 0.0)
        finally:
            env.close()

    def test_mapper_identity_never_leaks(self) -> None:
        env = self._bounded_sweep(band="hard")
        try:
            obs = env.reset(seed=7)
            offset = str(env._active_mapper_params["xor_offset"])
            info = env.step(Phase2Action(tool="dram.info", args={}))
            issue = env.step(Phase2Action(tool="dram.issue", args={
                "commands": [{"op": "RD", "addr": obs.metadata["candidates"][0]}]}))
            surfaces = [repr(obs.metadata), repr(info.metadata), repr(info.info),
                        repr(issue.feedback), repr(issue.metadata)]
            for blob in surfaces:
                for banned in ("RoBaRaCoChRowXOR", "xor_offset", "RoBaRaCoCh", "MOP4CLXOR", "addr_mapper"):
                    self.assertNotIn(banned, blob)
            # The secret offset value must not surface either (checked as a token).
            for blob in surfaces:
                self.assertNotIn(f"'xor_offset': {offset}", blob)
        finally:
            env.close()

    def test_physical_target_fails_closed_under_secret_mapper(self) -> None:
        # Defensive guard: the Python RoBaRaCoCh projection is wrong (and bank-secret
        # leaking) under a secret mapper, so physical disclosure fails closed.
        env = self._bounded_sweep()
        try:
            from rowhammer_env.tasks.compiler import TaskConfigError
            with self.assertRaises(TaskConfigError):
                env._physical_target(env._compiled.target_addr)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
