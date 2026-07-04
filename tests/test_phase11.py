from __future__ import annotations

import base64
import pathlib
import unittest

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv
from rowhammer_env.disturbance import DisturbanceEngine
from rowhammer_env.geometry import Geometry


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"

# A DDR4_8Gb_x8 INFO payload as the worker reports it (see IssuedEventRecorder).
DDR4_INFO = {
    "standard": "DDR4",
    "tx_bytes": 64,
    "prefetch": 8,
    "channel_width": 64,
    "level_names": ["Channel", "Rank", "BankGroup", "Bank", "Row", "Column"],
    "level_sizes": [1, 1, 4, 4, 65536, 1024],
}


def engine(**kwargs) -> DisturbanceEngine:
    return DisturbanceEngine(geometry=Geometry(DDR4_INFO), seed=11, **kwargs)


def act(row: int, *, bank: int = 0, bankgroup: int = 0) -> dict:
    return {"op": "ACT", "channel": 0, "rank": 0, "bankgroup": bankgroup, "bank": bank, "row": row, "row_hit": False}


class GeometryTests(unittest.TestCase):
    def test_row_stride_matches_robaracoch_layout(self) -> None:
        geo = Geometry(DDR4_INFO)
        # Col(7 bits after prefetch) + Rank(0) + BankGroup(2) + Bank(2) below Row,
        # above a 6-bit (64 B) transaction offset -> 2**17.
        self.assertEqual(geo.row_stride, 131072)
        self.assertEqual(geo.tx_bytes, 64)

    def test_non_power_of_two_rejected(self) -> None:
        bad = dict(DDR4_INFO, tx_bytes=48)
        with self.assertRaises(ValueError):
            Geometry(bad)


class ExposureModelTests(unittest.TestCase):
    """Engine-level checks with synthetic issued events (no worker required)."""

    def test_row_hits_without_acts_do_not_hammer(self) -> None:
        eng = engine()
        # Many reads that all hit an open row emit no ACT -> no exposure at all.
        for _ in range(5000):
            eng.consume([], {"op": "RD", "addr": eng.target_addr - eng.row_bytes, "size": 64})
        self.assertEqual(eng.victims, {})
        self.assertEqual(eng.flips, {})

    def test_double_sided_acts_flip_target(self) -> None:
        eng = engine()
        left_addr = eng.target_addr - eng.row_bytes
        right_addr = eng.target_addr + eng.row_bytes
        target_row = eng.known_target_row
        pairs = eng.known_threshold // 2
        for _ in range(pairs):
            eng.consume([act(target_row - 1)], {"op": "RD", "addr": left_addr, "size": 64})
            eng.consume([act(target_row + 1)], {"op": "RD", "addr": right_addr, "size": 64})
        victim = eng.victims[(0, 0, 0, 0, target_row)]
        self.assertEqual((victim.left, victim.right), (pairs, pairs))
        self.assertTrue(victim.flipped)
        self.assertEqual(eng.flips.get(eng.target_addr), 0)

    def test_single_sided_uses_the_higher_single_stratum_threshold(self) -> None:
        # P14: single-sided hammering is admitted but governed by the single
        # stratum's (much higher) hcfirst, not the double-sided condition. Below
        # that threshold the victim does not flip; crossing it flips.
        eng = engine()
        left_addr = eng.target_addr - eng.row_bytes
        target_key = (0, 0, 0, 0, eng.known_target_row)
        single_threshold = eng.known_single_threshold
        self.assertGreater(single_threshold, eng.known_threshold)
        for _ in range(single_threshold - 1):
            eng.consume([act(eng.known_target_row - 1)], {"op": "RD", "addr": left_addr, "size": 64})
        self.assertFalse(eng.victims[target_key].flipped)
        self.assertEqual(eng.flips, {})
        eng.consume([act(eng.known_target_row - 1)], {"op": "RD", "addr": left_addr, "size": 64})
        self.assertTrue(eng.victims[target_key].flipped)

    def test_exposure_keyed_by_decoded_bank(self) -> None:
        eng = engine()
        # Same row index in a different bank must be a distinct victim.
        eng.consume([act(9, bank=0)], {"op": "RD", "addr": 9 * eng.row_bytes, "size": 64})
        eng.consume([act(9, bank=1)], {"op": "RD", "addr": 9 * eng.row_bytes, "size": 64})
        self.assertIn((0, 0, 0, 0, 10), eng.victims)  # neighbour of bank-0 aggressor
        self.assertIn((0, 0, 0, 1, 10), eng.victims)  # neighbour of bank-1 aggressor

    def test_write_restores_and_refresh_does_not_correct_flips(self) -> None:
        eng = engine()
        addr = eng.target_addr
        eng.flips[addr] = 0
        # A refresh rewrites the disturbed value it reads: it never corrects an
        # existing flip (test D10), regardless of how many refreshes pass.
        for _ in range(eng.refresh_window + 1):
            eng.consume([{"op": "REFab", "row": -1}], {"op": "WAIT", "addr": 0, "size": 0})
        self.assertEqual(eng.flips.get(addr), 0)
        # A write to the region restores it.
        eng.consume([act(eng.known_target_row)], {"op": "WR", "addr": addr, "size": 1})
        self.assertNotIn(addr, eng.flips)

    def test_sampled_thresholds_are_deterministic(self) -> None:
        a = engine()
        b = engine()
        a.consume([act(200)], {"op": "RD", "addr": 200 * a.row_bytes, "size": 64})
        b.consume([act(200)], {"op": "RD", "addr": 200 * b.row_bytes, "size": 64})
        self.assertEqual(
            {k: v.threshold for k, v in a.victims.items()},
            {k: v.threshold for k, v in b.victims.items()},
        )


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class WorkerEventStreamTests(unittest.TestCase):
    """The real worker must surface decoded, row-buffer-aware issued commands."""

    def read(self, env: RowHammerDisturbanceEnv, addr: int):
        return env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": 1}))

    def test_worker_reports_geometry(self) -> None:
        env = RowHammerDisturbanceEnv()
        env.reset(seed=11)
        self.assertEqual(env.disturbance.row_bytes, 131072)
        self.assertEqual(env.disturbance.tx_bytes, 64)
        env.close()

    def test_read_emits_decoded_act(self) -> None:
        env = RowHammerDisturbanceEnv()
        env.reset(seed=11)
        row = 5
        obs = self.read(env, row * env.disturbance.row_bytes)
        events = obs.feedback["trace_tail"]
        acts = [e for e in events if e["op"] == "ACT"]
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0]["row"], row)
        self.assertEqual(acts[0]["bank"], 0)
        self.assertFalse(acts[0]["row_hit"])
        rds = [e for e in events if e["op"] == "RD"]
        self.assertTrue(rds and rds[-1]["row_hit"])
        env.close()

    def test_repeated_read_is_a_row_hit_without_act(self) -> None:
        env = RowHammerDisturbanceEnv()
        env.reset(seed=11)
        addr = 7 * env.disturbance.row_bytes
        self.read(env, addr)  # opens the row (ACT)
        obs = self.read(env, addr)  # same row -> hit, no ACT
        events = obs.feedback["trace_tail"]
        self.assertFalse([e for e in events if e["op"] == "ACT"])
        self.assertTrue([e for e in events if e["op"] == "RD" and e["row_hit"]])
        env.close()


if __name__ == "__main__":
    unittest.main()
