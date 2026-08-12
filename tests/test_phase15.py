"""Phase 15 — standards & profile generalization (defect F; IMPLEMENTATION_PLAN_V2 P15).

Engine-level checks that the disturbance engine is standard-generic: geometry is
derived from the real ``DRAMSpec`` for any standard (DDR4/DDR5/HBM2), the
per-standard adapter drives blast/refresh/impl selection, the profile schema is
versioned (v2) and backward-compatible, a profile is never pooled across
standards, and the HBM2 profile/source stays deferred and fails closed. The
worker path is covered by ``scripts/verify_phase15.py``; these run in the
source-free unit gate.

The DDR5/HBM2 INFO payloads below are the *real* geometries the corresponding
Ramulator configs report (captured from ``build/phase2/p2_external_{ddr5,hbm2}``),
not fabricated numbers.
"""

from __future__ import annotations

import copy
import unittest

from profile_builder.errors import SourceUnavailable
from profile_builder.ingest import hbm2 as hbm2_ingest
from profile_builder.standards import (
    STANDARD_FACTS,
    SUPPORTED_STANDARDS,
    UnsupportedStandard,
    as_profile_blocks,
    facts_for,
)
from rowhammer_env import disturbance as disturbance_mod
from rowhammer_env.disturbance import DisturbanceEngine
from rowhammer_env.geometry import Geometry
from rowhammer_env.tools.addressing import AddressMapper
from rowhammer_env.profiles import ADMITTED_PROFILES, DEFERRED_PROFILES, load_profile
from rowhammer_env.standards import StandardModel, ramulator_impl_for


DDR4_INFO = {
    "standard": "DDR4", "tx_bytes": 64, "prefetch": 8, "channel_width": 64,
    "level_names": ["Channel", "Rank", "BankGroup", "Bank", "Row", "Column"],
    "level_sizes": [1, 1, 4, 4, 65536, 1024],
}
DDR5_INFO = {
    "standard": "DDR5", "tx_bytes": 64, "prefetch": 16, "channel_width": 32,
    "level_names": ["Channel", "Rank", "BankGroup", "Bank", "Row", "Column"],
    "level_sizes": [1, 1, 8, 4, 65536, 1024],
}
HBM2_INFO = {
    "standard": "HBM2", "tx_bytes": 32, "prefetch": 4, "channel_width": 64,
    "level_names": ["Channel", "PseudoChannel", "BankGroup", "Bank", "Row", "Column"],
    "level_sizes": [1, 2, 4, 4, 65536, 128],
}


def encoder(info: dict):
    """Row->address resolution for a worker-free engine (public mapper only)."""
    return AddressMapper(Geometry(info)).encode


class GeometryStrideTests(unittest.TestCase):
    """row_stride is derived from the real level layout for any standard."""

    def test_ddr4_stride_unchanged(self) -> None:
        self.assertEqual(Geometry(DDR4_INFO).row_stride, 131072)

    def test_hbm2_stride_includes_pseudo_channel(self) -> None:
        # HBM2 has no Rank but a PseudoChannel level below Row; the generic stride
        # must count it (would be wrong if the DDR "rank/bankgroup/bank" set leaked).
        self.assertEqual(Geometry(HBM2_INFO).row_stride, 32768)

    def test_ddr5_stride_derived(self) -> None:
        self.assertEqual(Geometry(DDR5_INFO).row_stride, 131072)


class StandardModelTests(unittest.TestCase):
    """The standard adapter derives the right parameters per standard."""

    def test_ddr4_adapter(self) -> None:
        sm = StandardModel.from_geometry(Geometry(DDR4_INFO))
        self.assertEqual(sm.blast, ((1, 1.0),))
        self.assertEqual(sm.refresh_window, 8192)
        self.assertEqual(sm.ramulator_impl(), ("DDR4", "GenericDDR"))
        self.assertFalse(sm.facts.pseudo_channel)

    def test_hbm2_adapter_dimensions(self) -> None:
        sm = StandardModel.from_geometry(Geometry(HBM2_INFO))
        self.assertEqual(sm.ramulator_impl(), ("HBM2", "GenericDDR"))
        self.assertTrue(sm.facts.pseudo_channel)
        self.assertTrue(sm.facts.on_die_ecc)
        self.assertTrue(sm.facts.die_stacking)
        self.assertEqual(sm.row_bytes, 32768)

    def test_half_double_extends_blast_when_requested(self) -> None:
        base = StandardModel.from_geometry(Geometry(DDR4_INFO))
        hd = StandardModel.from_geometry(Geometry(DDR4_INFO), half_double=True)
        self.assertEqual(base.blast, ((1, 1.0),))
        self.assertEqual({d for d, _ in hd.blast}, {1, 2})

    def test_geometry_dimension_mismatch_fails_closed(self) -> None:
        # A geometry whose levels contradict the standard's dimensions is rejected
        # (e.g. claiming DDR4 but exposing a PseudoChannel level).
        bad = dict(HBM2_INFO, standard="DDR4")
        with self.assertRaises(UnsupportedStandard):
            StandardModel.from_geometry(Geometry(bad))

    def test_unknown_standard_fails_closed(self) -> None:
        with self.assertRaises(UnsupportedStandard):
            facts_for("GDDR6")
        with self.assertRaises(UnsupportedStandard):
            ramulator_impl_for("GDDR6")


class EngineStandardTests(unittest.TestCase):
    """The engine is parameterized by standard, with no DDR4 hard-gate."""

    def test_ddr4_engine_uses_adapter(self) -> None:
        eng = DisturbanceEngine(geometry=Geometry(DDR4_INFO), row_encoder=encoder(DDR4_INFO), seed=15)
        self.assertEqual(eng.standard, "DDR4")
        self.assertEqual(tuple(eng.blast), ((1, 1.0),))
        self.assertEqual(eng.refresh_window, 8192)

    def test_no_pooling_ddr4_profile_on_hbm2_geometry(self) -> None:
        # The DDR4-fitted profile must never run on an HBM2 geometry.
        with self.assertRaises(ValueError) as ctx:
            DisturbanceEngine(geometry=Geometry(HBM2_INFO), row_encoder=encoder(HBM2_INFO), seed=15)
        self.assertTrue(str(ctx.exception).startswith("PROFILE_REJECTED:"))

    def test_no_pooling_ddr4_profile_on_ddr5_geometry(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            DisturbanceEngine(geometry=Geometry(DDR5_INFO), row_encoder=encoder(DDR5_INFO), seed=15)
        self.assertTrue(str(ctx.exception).startswith("PROFILE_REJECTED:"))

    def test_v1_profile_backward_compatible(self) -> None:
        # A legacy v1 package (no topology/refresh blocks) must give identical
        # engine behaviour by falling back to the standard adapter's values.
        geo = Geometry(DDR4_INFO)
        v2 = DisturbanceEngine(geometry=geo, row_encoder=AddressMapper(geo).encode, seed=15)

        legacy = copy.deepcopy(load_profile("ddr4_vts25_v1"))
        for key in ("schema_version", "topology", "refresh", "standard_dimensions"):
            legacy.pop(key, None)
        original = disturbance_mod.load_profile
        disturbance_mod.load_profile = lambda _pid: copy.deepcopy(legacy)
        try:
            v1 = DisturbanceEngine(geometry=geo, row_encoder=AddressMapper(geo).encode, seed=15)
        finally:
            disturbance_mod.load_profile = original

        self.assertEqual(tuple(v1.blast), tuple(v2.blast))
        self.assertEqual(v1.refresh_window, v2.refresh_window)
        self.assertEqual(v1.known_threshold, v2.known_threshold)


class SchemaV2Tests(unittest.TestCase):
    """The committed profile is schema v2 and carries consistent standard blocks."""

    def test_committed_profile_is_v2_with_standard_blocks(self) -> None:
        profile = load_profile("ddr4_vts25_v1")
        self.assertEqual(str(profile["schema_version"]), "2")
        expected = as_profile_blocks(facts_for("DDR4"))
        for key in ("topology", "refresh", "standard_dimensions"):
            self.assertEqual(profile[key], expected[key])

    def test_ddr4_profile_declares_no_hbm_dimensions(self) -> None:
        dims = load_profile("ddr4_vts25_v1")["standard_dimensions"]
        self.assertFalse(dims["pseudo_channel"])
        self.assertFalse(dims["on_die_ecc"])
        self.assertFalse(dims["die_stacking"])

    def test_supported_standards_cover_ddr_and_hbm(self) -> None:
        self.assertIn("DDR4", SUPPORTED_STANDARDS)
        self.assertIn("DDR5", SUPPORTED_STANDARDS)
        self.assertIn("HBM2", SUPPORTED_STANDARDS)
        self.assertEqual(set(STANDARD_FACTS), set(SUPPORTED_STANDARDS))


class DeferredHbm2Tests(unittest.TestCase):
    """The HBM2 profile/source is deferred and fails closed (no parameter pooling)."""

    def test_hbm2_profile_not_admitted(self) -> None:
        self.assertNotIn("hbm2_read_disturbance", ADMITTED_PROFILES)
        self.assertIn("hbm2_read_disturbance", DEFERRED_PROFILES)

    def test_hbm2_profile_load_fails_closed(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            load_profile("hbm2_read_disturbance")
        self.assertTrue(str(ctx.exception).startswith("UNAVAILABLE_CAPABILITY:"))

    def test_hbm2_source_deferred_and_ingestion_fails_closed(self) -> None:
        self.assertFalse(hbm2_ingest.source_status().admitted)
        with self.assertRaises(SourceUnavailable):
            hbm2_ingest.require_admitted()
        with self.assertRaises(SourceUnavailable):
            hbm2_ingest.load_rd()


class NoChipPoolingTests(unittest.TestCase):
    """Each DDR4 chip family keeps an independent held-out split (no pooling)."""

    def test_family_holdout_disjoint_from_train(self) -> None:
        profile = load_profile("ddr4_vts25_v1")
        for family, fam in profile["fit"]["families"].items():
            self.assertFalse(
                set(fam["chips_holdout"]) & set(fam["chips_train"]),
                f"{family} pools held-out chips into training",
            )


if __name__ == "__main__":
    unittest.main()
