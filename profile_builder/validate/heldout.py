"""Held-out statistical validation of the fitted model.

The fit is built from training units only; here it is scored against units it never
saw (TEST_PLAN P3/P4), separated into two tiers that test different generalization:

- Within-module (row holdout): unseen victim rows of the training chips. The fitted
  lognormal must describe these well, so both a tight median-ratio gate and a strict
  Kolmogorov-Smirnov gate apply.
- Cross-module (chip holdout): entirely unseen chips. Distribution shape legitimately
  shifts between modules, so only central tendency (a looser median-ratio gate) and
  qualitative structure (single>double ordering, dominant bitflip direction for
  directionally-biased families) are gated; the cross-module KS distance is reported
  as a diagnostic only.

These are sanity gates, not tight scientific bounds. If any gated check fails, the
build fails closed and the profile is not admitted.
"""

from __future__ import annotations

import numpy as np

from ..canonicalize import CanonicalTables
from ..fit.distributions import ks_statistic
from ..fit.profile import AGGR_CLASSES, DATA_PATTERNS, stratum_key
from .split import HOLDOUT_CHIPS, partition_rd

DEFAULT_GATES = {
    "row_ks_gate": 0.30,
    "row_median_ratio_gate": 1.5,
    "chip_median_ratio_gate": 2.5,
}


def _hc(records, *, family=None, aggr_class=None, pattern=None):
    return np.array(
        [
            r.hc
            for r in records
            if (family is None or r.family == family)
            and (aggr_class is None or r.aggr_class == aggr_class)
            and (pattern is None or r.pattern == pattern)
            and r.num_bitflips > 0
        ],
        dtype=float,
    )


def _within(lo: float, hi: float, ratio: float) -> bool:
    return lo <= ratio <= hi


def build_report(profile_fit: dict, tables: CanonicalTables, **gates: float) -> dict:
    g = {**DEFAULT_GATES, **gates}
    split = partition_rd(tables.rd["rd_hcf"])
    row_holdout = [r for r in split.holdout if r.chip not in HOLDOUT_CHIPS]
    chip_holdout = [r for r in split.holdout if r.chip in HOLDOUT_CHIPS]
    train_chips = {r.chip for r in split.train}

    checks: list[dict] = []

    def add(passed: bool, **fields) -> None:
        checks.append({**fields, "passed": bool(passed)})

    for family, fam_fit in profile_fit["families"].items():
        biased = fam_fit["direction"]["biased"]
        dom = fam_fit["direction"]["dominant_pattern"]

        for aggr_class in AGGR_CLASSES:
            for pattern in DATA_PATTERNS:
                key = stratum_key(aggr_class, pattern)
                ln = fam_fit["strata"][key]["hcfirst_lognormal"]

                row = _hc(row_holdout, family=family, aggr_class=aggr_class, pattern=pattern)
                if row.size >= 2:
                    ratio = float(np.median(row)) / ln["median"]
                    add(
                        _within(1.0 / g["row_median_ratio_gate"], g["row_median_ratio_gate"], ratio),
                        family=family, stratum=key, tier="within_module",
                        check="median_ratio", ratio=ratio, gate=g["row_median_ratio_gate"],
                        n_holdout=int(row.size),
                    )
                    ks = ks_statistic(row, ln["mu"], ln["sigma"])
                    add(
                        ks <= g["row_ks_gate"],
                        family=family, stratum=key, tier="within_module",
                        check="ks", ks=ks, gate=g["row_ks_gate"], n_holdout=int(row.size),
                    )

                chip = _hc(chip_holdout, family=family, aggr_class=aggr_class, pattern=pattern)
                if chip.size >= 2:
                    ratio = float(np.median(chip)) / ln["median"]
                    add(
                        _within(1.0 / g["chip_median_ratio_gate"], g["chip_median_ratio_gate"], ratio),
                        family=family, stratum=key, tier="cross_module",
                        check="median_ratio", ratio=ratio, gate=g["chip_median_ratio_gate"],
                        n_holdout=int(chip.size),
                    )
                    # Cross-module KS is a diagnostic only; shape shift is expected.
                    checks.append({
                        "family": family, "stratum": key, "tier": "cross_module",
                        "check": "ks_diagnostic", "ks": ks_statistic(chip, ln["mu"], ln["sigma"]),
                        "n_holdout": int(chip.size), "diagnostic": True, "passed": True,
                    })

        # Qualitative structure on each available tier.
        for tier, subset in (("within_module", row_holdout), ("cross_module", chip_holdout)):
            for pattern in DATA_PATTERNS:
                single = _hc(subset, family=family, aggr_class="single", pattern=pattern)
                double = _hc(subset, family=family, aggr_class="double", pattern=pattern)
                if single.size and double.size:
                    add(
                        float(np.median(single)) > float(np.median(double)),
                        family=family, stratum=f"single_gt_double|{pattern}", tier=tier,
                        check="ordering", single_median=float(np.median(single)),
                        double_median=float(np.median(double)),
                    )

            ones = _hc(subset, family=family, aggr_class="single", pattern="all_ones")
            zeros = _hc(subset, family=family, aggr_class="single", pattern="all_zeros")
            if ones.size and zeros.size:
                observed = "all_ones" if np.median(ones) < np.median(zeros) else "all_zeros"
                # Direction is gated only for directionally-biased families.
                checks.append({
                    "family": family, "stratum": "direction", "tier": tier, "check": "direction",
                    "fitted_dominant_pattern": dom, "observed_dominant_pattern": observed,
                    "bias_strength": fam_fit["direction"]["bias_strength"], "gated": biased,
                    "diagnostic": not biased,
                    "passed": (observed == dom) if biased else True,
                })

    leakage_free = train_chips.isdisjoint(HOLDOUT_CHIPS)
    gated = [c for c in checks if not c.get("diagnostic")]
    passed = leakage_free and all(c["passed"] for c in gated)
    return {
        "gates": g,
        "holdout_chips": sorted(HOLDOUT_CHIPS),
        "train_chip_holdout_chip_disjoint": leakage_free,
        "n_checks": len(checks),
        "n_gated": len(gated),
        "n_failed": sum(1 for c in gated if not c["passed"]),
        "passed": bool(passed),
        "checks": checks,
    }
