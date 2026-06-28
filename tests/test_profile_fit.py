"""Fit/ingest tests that require the fetched source data.

Covers TEST_PLAN P2 (ingestion reproduces source counts), P3 (held-out units are
never used in fitting; train/hold-out chips are disjoint), and P4 (held-out gates
pass on the real data and have teeth against a perturbation), plus the determinism
acceptance check (a rebuild reproduces the committed package).

These skip when the source data has not been fetched; the full check is the
``scripts/verify_phase3.py`` admission gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from profile_builder.canonicalize import (
    CanonicalTables,
    build_tables,
    canonical_counts,
    source_row_counts,
)
from profile_builder.fit.profile import build_profile_fit
from profile_builder.ingest import AGGR_CLASS, PATTERNS, family_of
from profile_builder.package.build import build
from profile_builder.paths import DATA_DIR, OUT_DIR
from profile_builder.validate.heldout import build_report
from profile_builder.validate.split import HOLDOUT_CHIPS, is_holdout_row, partition_rd

DATA_AVAILABLE = DATA_DIR.is_dir() and any(DATA_DIR.glob("*_rd_hcf.csv"))


def _retable(tables: CanonicalTables, hcf) -> CanonicalTables:
    return CanonicalTables(rd={**tables.rd, "rd_hcf": hcf}, ds_ber=tables.ds_ber)


@unittest.skipUnless(DATA_AVAILABLE, "source data not fetched; run scripts/fetch_phase3_sources.py")
class IngestTests(unittest.TestCase):
    def test_canonical_counts_reproduce_source_rows(self) -> None:  # P2
        tables = build_tables()
        self.assertEqual(canonical_counts(tables), source_row_counts())

    def test_normalization_mapping(self) -> None:  # P2
        tables = build_tables()
        for r in tables.rd["rd_hcf"]:
            self.assertIn(r.pattern, {p for p, _ in PATTERNS.values()})
            self.assertIn(r.aggr_class, set(AGGR_CLASS.values()))
            self.assertEqual(r.family, family_of(r.chip))
            expected_dir = "1->0" if r.pattern == "all_ones" else "0->1"
            self.assertEqual(r.direction, expected_dir)


@unittest.skipUnless(DATA_AVAILABLE, "source data not fetched; run scripts/fetch_phase3_sources.py")
class HeldOutTests(unittest.TestCase):
    def test_holdout_units_not_used_in_fit(self) -> None:  # P3
        tables = build_tables()

        def corrupt(r):
            held = r.chip in HOLDOUT_CHIPS or is_holdout_row(r.chip, r.victim_row)
            return dataclasses.replace(r, hc=r.hc * 1000) if held else r

        corrupted = _retable(tables, [corrupt(r) for r in tables.rd["rd_hcf"]])
        # The fit consumes only training units, so corrupting every held-out unit
        # must not change any fitted parameter.
        self.assertEqual(build_profile_fit(tables), build_profile_fit(corrupted))

    def test_train_and_holdout_chips_are_disjoint(self) -> None:  # P3
        split = partition_rd(build_tables().rd["rd_hcf"])
        train_chips = {r.chip for r in split.train}
        self.assertTrue(train_chips.isdisjoint(HOLDOUT_CHIPS))
        # Held-out chips appear only on the held-out side.
        self.assertEqual({r.chip for r in split.holdout} & HOLDOUT_CHIPS, HOLDOUT_CHIPS)

    def test_distribution_gates_pass(self) -> None:  # P4
        tables = build_tables()
        report = build_report(build_profile_fit(tables), tables)
        self.assertTrue(report["passed"], report["n_failed"])

    def test_gates_detect_within_module_misfit(self) -> None:  # P4 negative control
        tables = build_tables()
        fit = build_profile_fit(tables)

        def inflate(r):
            row_held = r.chip not in HOLDOUT_CHIPS and is_holdout_row(r.chip, r.victim_row)
            return dataclasses.replace(r, hc=r.hc * 5) if row_held else r

        corrupted = _retable(tables, [inflate(r) for r in tables.rd["rd_hcf"]])
        report = build_report(fit, corrupted)
        self.assertFalse(report["passed"])


@unittest.skipUnless(DATA_AVAILABLE, "source data not fetched; run scripts/fetch_phase3_sources.py")
class DeterminismTests(unittest.TestCase):
    def test_rebuild_reproduces_committed_profile(self) -> None:  # acceptance #7
        committed = (OUT_DIR / "profile.json").read_bytes()
        rebuilt = build(write=False)
        self.assertEqual(rebuilt["payload_sha256"], hashlib.sha256(committed).hexdigest())


if __name__ == "__main__":
    unittest.main()
