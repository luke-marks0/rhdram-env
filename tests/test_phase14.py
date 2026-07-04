"""Phase 14 — disturbance fidelity (defects B, C, E; TEST_PLAN D4–D10).

These are engine-level checks that drive the *real* ``DisturbanceEngine`` (the
admitted ``ddr4_vts25_v1`` profile, loaded and signature-verified) with the same
decoded issued-event shape the worker emits. No worker build is required, so they
run in the source-free unit gate; the end-to-end worker path is covered by
``scripts/verify_phase14.py``.
"""

from __future__ import annotations

import math
import statistics
import unittest

from rowhammer_env.disturbance import DisturbanceEngine
from rowhammer_env.geometry import Geometry


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
    return DisturbanceEngine(geometry=Geometry(DDR4_INFO), seed=14, **kwargs)


def act(row: int, *, bank: int = 0, bankgroup: int = 0, clk: int = 0) -> dict:
    return {
        "op": "ACT",
        "channel": 0,
        "rank": 0,
        "bankgroup": bankgroup,
        "bank": bank,
        "row": row,
        "row_hit": False,
        "clk": clk,
    }


def pre(*, bank: int = 0, bankgroup: int = 0, clk: int = 0) -> dict:
    return {"op": "PRE", "channel": 0, "rank": 0, "bankgroup": bankgroup, "bank": bank, "clk": clk}


def refab() -> dict:
    return {"op": "REFab", "channel": 0, "rank": 0, "row": -1, "clk": 0}


def rd(addr: int) -> dict:
    return {"op": "RD", "addr": addr, "size": 64}


def hammer_double(eng: DisturbanceEngine, pairs: int) -> None:
    """Double-sided hammer of the known target for ``pairs`` activations per side."""
    left = eng.target_addr - eng.row_bytes
    right = eng.target_addr + eng.row_bytes
    tr = eng.known_target_row
    for _ in range(pairs):
        eng.consume([act(tr - 1)], rd(left))
        eng.consume([act(tr + 1)], rd(right))


class KnownFlipControlTests(unittest.TestCase):
    """D4 no-flip control and D5 known-flip fixture across fixed seeds."""

    def test_d5_known_flip_fixture(self) -> None:
        for seed in (1, 7, 14, 99):
            eng = DisturbanceEngine(geometry=Geometry(DDR4_INFO), seed=seed)
            hammer_double(eng, eng.known_threshold // 2)
            victim = eng.victims[(0, 0, 0, 0, eng.known_target_row)]
            self.assertTrue(victim.flipped, f"seed {seed} did not flip at threshold")
            # all-zeros region, 0->1, bit 0 -> the target byte reads 1.
            self.assertEqual(eng.flips.get(eng.target_addr), 0)
            self.assertEqual(eng.apply(eng.target_addr, b"\x00")[0], 1)

    def test_d4_below_threshold_no_flip(self) -> None:
        for seed in (1, 7, 14, 99):
            eng = DisturbanceEngine(geometry=Geometry(DDR4_INFO), seed=seed)
            hammer_double(eng, eng.known_threshold // 2 - 1)
            self.assertEqual(eng.flips, {})
            self.assertFalse(eng.victims[(0, 0, 0, 0, eng.known_target_row)].flipped)


class SingleVsDoubleTests(unittest.TestCase):
    """D6 — single/double-sided exposure differs according to the profile."""

    def test_single_threshold_far_above_double(self) -> None:
        eng = engine()
        self.assertGreater(eng.known_single_threshold, eng.known_threshold)
        # The profile's single_to_double ratio for this stratum is > 1.
        ratio = eng.known_single_threshold / eng.known_threshold
        self.assertGreater(ratio, 1.5)

    def test_double_sided_flips_with_far_fewer_activations(self) -> None:
        # Double-sided crosses at known_threshold; single-sided at the (much
        # larger) single threshold — same row, different adjacency, different
        # exposure required.
        eng = engine()
        target_key = (0, 0, 0, 0, eng.known_target_row)
        hammer_double(eng, eng.known_threshold // 2)
        self.assertTrue(eng.victims[target_key].flipped)

        single = engine()
        left = single.target_addr - single.row_bytes
        for _ in range(single.known_threshold):  # far past the double threshold
            single.consume([act(single.known_target_row - 1)], rd(left))
        # Single-sided has not yet reached its own (higher) threshold.
        self.assertFalse(single.victims[target_key].flipped)


class DirectionAndMultiplicityTests(unittest.TestCase):
    """Direction from the stored pattern; bit multiplicity grows with exposure."""

    def test_direction_follows_data_pattern(self) -> None:
        # Fresh (all-zeros) victim flips 0->1.
        eng = engine()
        hammer_double(eng, eng.known_threshold // 2)
        self.assertEqual(eng.victims[(0, 0, 0, 0, eng.known_target_row)].direction, "0->1")

        # A victim region written all-ones flips 1->0.
        eng2 = engine()
        row = 4000
        victim_addr = row * eng2.row_bytes
        eng2.note_write(victim_addr, b"\xff" * 64)
        agg_left = victim_addr - eng2.row_bytes
        agg_right = victim_addr + eng2.row_bytes
        # Drive both neighbours until it flips.
        v = None
        for _ in range(200000):
            eng2.consume([act(row - 1)], rd(agg_left))
            eng2.consume([act(row + 1)], rd(agg_right))
            v = eng2.victims[(0, 0, 0, 0, row)]
            if v.flipped:
                break
        assert v is not None
        self.assertTrue(v.flipped)
        self.assertEqual(v.direction, "1->0")

    def test_multiplicity_grows_with_exposure(self) -> None:
        at_threshold = engine()
        hammer_double(at_threshold, at_threshold.known_threshold // 2)
        self.assertEqual(at_threshold.victims[(0, 0, 0, 0, at_threshold.known_target_row)].flipped_bits, 1)

        far_past = engine()
        hammer_double(far_past, far_past.known_threshold // 2 * 4)
        self.assertGreater(far_past.victims[(0, 0, 0, 0, far_past.known_target_row)].flipped_bits, 1)


class RowPressTests(unittest.TestCase):
    """D7 — open-row dwell reduces hcfirst, and only for RowPress profiles."""

    def _pairs_to_flip(self, eng: DisturbanceEngine, dwell: int) -> int:
        left = eng.target_addr - eng.row_bytes
        right = eng.target_addr + eng.row_bytes
        tr = eng.known_target_row
        key = (0, 0, 0, 0, tr)
        clk = 0
        for n in range(1, 20000):
            eng.consume([act(tr - 1, clk=clk)], rd(left))
            clk += dwell
            eng.consume([pre(clk=clk), act(tr + 1, clk=clk)], rd(right))
            clk += dwell
            eng.consume([pre(clk=clk)], rd(left))  # close the right aggressor
            clk += 10
            if eng.victims[key].flipped:
                return n
        raise AssertionError("victim never flipped")

    def test_rowpress_dwell_reduces_hammers_to_flip(self) -> None:
        short = self._pairs_to_flip(engine(), dwell=10)
        long = self._pairs_to_flip(engine(), dwell=200000)
        # RowPress makes each long-open activation worth many hammers.
        self.assertLess(long * 3, short)

    def test_rowpress_ignored_when_profile_unsupported(self) -> None:
        eng = engine()
        # Turn RowPress support off for the active family (a profile-shape
        # variation, exercising the not-supported branch): dwell must not help.
        eng.profile["fit"]["families"][eng.family]["rowpress"]["supported"] = False
        self.assertEqual(eng._rowpress_factor(200000), 1.0)


class RefreshDecayTests(unittest.TestCase):
    """D8 — refresh decays exposure; D10 — flips persist through refresh."""

    def test_fast_burst_within_window_flips(self) -> None:
        eng = engine()
        # A handful of refreshes (well under a window) does not reset exposure.
        left = eng.target_addr - eng.row_bytes
        right = eng.target_addr + eng.row_bytes
        tr = eng.known_target_row
        for _ in range(eng.known_threshold // 2):
            eng.consume([act(tr - 1)], rd(left))
            eng.consume([refab(), act(tr + 1)], rd(right))
        self.assertTrue(eng.victims[(0, 0, 0, 0, tr)].flipped)

    def test_full_refresh_window_resets_exposure(self) -> None:
        eng = engine()
        left = eng.target_addr - eng.row_bytes
        right = eng.target_addr + eng.row_bytes
        tr = eng.known_target_row
        key = (0, 0, 0, 0, tr)
        for _ in range(eng.known_threshold // 2 - 5):
            eng.consume([act(tr - 1)], rd(left))
            eng.consume([act(tr + 1)], rd(right))
        self.assertGreater(eng.victims[key].left, 0)
        # A full refresh window elapses -> the row is refreshed, exposure clears.
        for _ in range(eng.refresh_window):
            eng.consume([refab()], {"op": "WAIT", "addr": 0, "size": 0})
        self.assertEqual(eng.victims[key].left, 0)
        self.assertEqual(eng.victims[key].right, 0)
        # And the accumulated-but-reset row can no longer flip on the leftover hammer.
        for _ in range(5):
            eng.consume([act(tr - 1)], rd(left))
            eng.consume([act(tr + 1)], rd(right))
        self.assertFalse(eng.victims[key].flipped)

    def test_flip_persists_through_refresh_until_write(self) -> None:
        eng = engine()
        hammer_double(eng, eng.known_threshold // 2)
        self.assertEqual(eng.flips.get(eng.target_addr), 0)
        for _ in range(eng.refresh_window * 2):
            eng.consume([refab()], {"op": "WAIT", "addr": 0, "size": 0})
        self.assertEqual(eng.flips.get(eng.target_addr), 0)  # refresh never corrects
        eng.consume([act(eng.known_target_row)], {"op": "WR", "addr": eng.target_addr, "size": 1})
        self.assertNotIn(eng.target_addr, eng.flips)


class OracleReferenceTests(unittest.TestCase):
    """D8 — the Python oracle matches Ramulator's OracleRH on a differential trace.

    ``_OracleRHReference`` is a direct transcription of the ACT/VRR logic in
    ``third_party/ramulator2/src/ramulator/controller/plugin/impl/oracle_rh.cpp``
    (per-aggressor activation counter; victim-row refresh at ``tRH``). Driving it
    and the production ``mitigation="oracle"`` engine with the *same* issued-ACT
    stream must agree on when protection fires and leave the target unflipped.
    """

    class _OracleRHReference:
        def __init__(self, tRH: int) -> None:
            self.tRH = tRH
            self.counts: dict[tuple[int, int], int] = {}
            self.vrr = 0

        def on_act(self, bank: int, row: int) -> None:
            key = (bank, row)
            self.counts[key] = self.counts.get(key, 0) + 1
            if self.counts[key] >= self.tRH:
                self.counts[key] = 0
                self.vrr += 1

    def test_oracle_matches_reference_and_protects(self) -> None:
        eng = engine(mitigation="oracle")
        ref = self._OracleRHReference(eng.tRH)
        left = eng.target_addr - eng.row_bytes
        right = eng.target_addr + eng.row_bytes
        tr = eng.known_target_row
        produced = 0
        for _ in range(eng.known_threshold):
            produced += eng.consume([act(tr - 1)], rd(left)).oracle_refreshes
            ref.on_act(0, tr - 1)
            produced += eng.consume([act(tr + 1)], rd(right)).oracle_refreshes
            ref.on_act(0, tr + 1)
        self.assertGreater(produced, 0)
        self.assertEqual(produced, ref.vrr)  # differential trace: identical firings
        self.assertNotIn(eng.target_addr, eng.flips)  # target protected

    def test_oracle_vrr_count_follows_tRH(self) -> None:
        # OracleRH fires once per tRH activations of each aggressor.
        eng = engine(mitigation="oracle", mitigation_params={"tRH": 500})
        left = eng.target_addr - eng.row_bytes
        tr = eng.known_target_row
        fired = 0
        for _ in range(2000):
            fired += eng.consume([act(tr - 1)], rd(left)).oracle_refreshes
        self.assertEqual(fired, 2000 // 500)


class TemperatureDomainTests(unittest.TestCase):
    """D9 — out-of-domain temperatures fail closed; supported ones are admitted."""

    def test_unsupported_temperature_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            engine(temperature=85)
        self.assertTrue(str(ctx.exception).startswith("PROFILE_REJECTED:"))

    def test_supported_temperature_admitted(self) -> None:
        self.assertEqual(engine(temperature=50).temperature, 50)


class StratumSelectionTests(unittest.TestCase):
    """Data pattern selects the profile stratum (SPEC §5.3)."""

    def test_written_pattern_selects_stratum(self) -> None:
        zeros = engine()
        ones = engine()
        row = 5000
        victim_addr = row * zeros.row_bytes
        ones.note_write(victim_addr, b"\xff" * 64)
        for eng in (zeros, ones):
            eng.consume([act(row - 1)], rd(victim_addr - eng.row_bytes))
        vz = zeros.victims[(0, 0, 0, 0, row)]
        vo = ones.victims[(0, 0, 0, 0, row)]
        self.assertEqual(vz.data_pattern, "all_zeros")
        self.assertEqual(vo.data_pattern, "all_ones")
        # Different strata -> generally different sampled thresholds.
        self.assertNotEqual(vz.threshold, vo.threshold)


class HierarchicalSamplingTests(unittest.TestCase):
    """Statistical validity — the sampler reproduces the profile's parameters."""

    def test_sampler_recovers_profile_parameters(self) -> None:
        family, stratum = "hisasa", "double|all_zeros"
        params = engine().profile["fit"]["families"][family]["strata"][stratum]["hcfirst_lognormal"]
        module_means: list[float] = []
        within_sigmas: list[float] = []
        all_logs: list[float] = []
        for m in range(60):
            eng = DisturbanceEngine(geometry=Geometry(DDR4_INFO), seed=5000 + m, family=family, stratum=stratum)
            logs = [math.log(eng._sample_threshold((0, 0, 0, 0, r), "double", "all_zeros")) for r in range(3, 400)]
            module_means.append(statistics.mean(logs))
            within_sigmas.append(statistics.pstdev(logs))
            all_logs.extend(logs)

        # Within-module spread recovers sigma_within; module-to-module spread
        # recovers sigma_between; the pooled median recovers exp(mu).
        self.assertAlmostEqual(statistics.mean(within_sigmas), params["sigma_within_chip"], delta=0.03)
        self.assertAlmostEqual(statistics.pstdev(module_means), params["sigma_between_chip"], delta=0.15)
        median_ratio = math.exp(statistics.median(all_logs)) / params["median"]
        self.assertLess(1.0 / 1.5, median_ratio)
        self.assertLess(median_ratio, 1.5)

    def test_thresholds_are_deterministic_per_seed(self) -> None:
        a = engine()
        b = engine()
        for r in (100, 250, 999):
            a.consume([act(r)], rd(r * a.row_bytes))
            b.consume([act(r)], rd(r * b.row_bytes))
        self.assertEqual(
            {k: (v.threshold, v.single_threshold) for k, v in a.victims.items()},
            {k: (v.threshold, v.single_threshold) for k, v in b.victims.items()},
        )


if __name__ == "__main__":
    unittest.main()
