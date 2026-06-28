"""Assemble the fitted read-disturbance model from canonical tables.

The fit consumes only the training partition (see :mod:`profile_builder.validate.split`).
For each family and stratum (aggressor class x data pattern) it fits a lognormal
HC-first model with a two-level variance decomposition so a consumer can sample a
persistent per-module offset and per-row threshold:

    log N_threshold(row) = mu + module_offset + row_eps
    module_offset ~ Normal(0, sigma_between_chip)
    row_eps       ~ Normal(0, sigma_within_chip)

It also records the single/double exposure ratio, the empirical bitflip-direction
dependence, the RowPress threshold reduction, a multiplicity (BER-vs-HC) table, and
the temperature domain.
"""

from __future__ import annotations

import math

import numpy as np

from ..canonicalize import CanonicalTables
from ..ingest import DsBerRecord, RdRecord
from ..validate.split import HOLDOUT_CHIPS, partition_rd
from .distributions import lognormal_fit, quantiles

AGGR_CLASSES = ("single", "double")
DATA_PATTERNS = ("all_ones", "all_zeros")
STRATA = tuple(f"{a}|{p}" for a in AGGR_CLASSES for p in DATA_PATTERNS)


def stratum_key(aggr_class: str, pattern: str) -> str:
    return f"{aggr_class}|{pattern}"


def _hc(records: list[RdRecord]) -> np.ndarray:
    return np.array([r.hc for r in records], dtype=float)


def _select(records, *, family=None, aggr_class=None, pattern=None, chip=None, flips=False):
    out = []
    for r in records:
        if family is not None and r.family != family:
            continue
        if aggr_class is not None and r.aggr_class != aggr_class:
            continue
        if pattern is not None and r.pattern != pattern:
            continue
        if chip is not None and r.chip != chip:
            continue
        if flips and r.num_bitflips <= 0:
            continue
        out.append(r)
    return out


def _fit_stratum(train_hcf: list[RdRecord], family: str, aggr_class: str, pattern: str) -> dict:
    recs = _select(train_hcf, family=family, aggr_class=aggr_class, pattern=pattern, flips=True)
    values = _hc(recs)
    fit = lognormal_fit(values)
    qs = quantiles(values)

    per_chip_mu: list[float] = []
    per_chip_sigma: list[float] = []
    for chip in sorted({r.chip for r in recs}):
        chip_vals = _hc(_select(recs, chip=chip))
        if chip_vals.size >= 2:
            logs = np.log(chip_vals[chip_vals > 0])
            per_chip_mu.append(float(logs.mean()))
            per_chip_sigma.append(float(logs.std(ddof=0)))

    sigma_between = float(np.std(per_chip_mu, ddof=1)) if len(per_chip_mu) >= 2 else 0.0
    sigma_within = float(np.median(per_chip_sigma)) if per_chip_sigma else fit["sigma"]

    return {
        "hcfirst_lognormal": {
            "mu": fit["mu"],
            "sigma": fit["sigma"],
            "sigma_within_chip": sigma_within,
            "sigma_between_chip": sigma_between,
            "median": fit["median"],
        },
        "quantiles": qs,
        "n_train": fit["n"],
        "n_chips": len(per_chip_mu),
    }


def _multiplicity(train_ber: list[RdRecord], family: str, aggr_class: str, pattern: str) -> list[list[float]]:
    recs = _select(train_ber, family=family, aggr_class=aggr_class, pattern=pattern)
    by_hc: dict[int, list[int]] = {}
    for r in recs:
        by_hc.setdefault(r.hc, []).append(r.num_bitflips)
    return [[float(hc), float(np.mean(flips))] for hc, flips in sorted(by_hc.items())]


def _rowpress(train_hcf, train_rp, family: str) -> dict:
    rp_flips = _select(train_rp, family=family, flips=True)
    supported = len(rp_flips) > 0
    operating_points = sorted({r.hc for r in _select(train_rp, family=family)})
    rp_single_ones = _hc(_select(train_rp, family=family, aggr_class="single", pattern="all_ones", flips=True))
    hcf_single_ones = _hc(_select(train_hcf, family=family, aggr_class="single", pattern="all_ones", flips=True))
    reduction = None
    if rp_single_ones.size and hcf_single_ones.size:
        reduction = float(np.median(hcf_single_ones) / np.median(rp_single_ones))
    return {
        "supported": supported,
        "operating_point_hc": [float(x) for x in operating_points],
        "rowhammer_to_rowpress_hc_reduction": reduction,
        "n_rowpress_flip_records": len(rp_flips),
    }


def _direction(train_hcf, family: str, bias_min_strength: float) -> dict:
    medians = {}
    for pattern in DATA_PATTERNS:
        vals = _hc(_select(train_hcf, family=family, aggr_class="single", pattern=pattern, flips=True))
        medians[pattern] = float(np.median(vals)) if vals.size else math.inf
    # Lower HC-first => more vulnerable => dominant observed flip direction.
    dominant_pattern = min(medians, key=medians.get)
    direction = {"all_ones": "1->0", "all_zeros": "0->1"}[dominant_pattern]
    lo, hi = sorted(medians.values())
    strength = float(hi / lo) if lo > 0 and math.isfinite(hi) else 1.0
    return {
        "single_median_hcfirst": {k: (None if math.isinf(v) else v) for k, v in medians.items()},
        "dominant_pattern": dominant_pattern,
        "dominant_flip_direction": direction,
        "bias_strength": strength,
        "biased": strength >= bias_min_strength,
    }


def _single_double_ratio(train_hcf, family: str) -> dict:
    out = {}
    for pattern in DATA_PATTERNS:
        single = _hc(_select(train_hcf, family=family, aggr_class="single", pattern=pattern, flips=True))
        double = _hc(_select(train_hcf, family=family, aggr_class="double", pattern=pattern, flips=True))
        out[pattern] = (
            float(np.median(single) / np.median(double)) if single.size and double.size else None
        )
    return out


def _temperature(ds_ber: list[DsBerRecord], family: str) -> dict:
    fam = [d for d in ds_ber if d.family == family]
    temps = sorted({d.temp for d in fam})
    per_temp_median = {}
    for t in temps:
        vals = np.array([d.hc for d in fam if d.temp == t and d.chip not in HOLDOUT_CHIPS], dtype=float)
        if vals.size:
            per_temp_median[str(t)] = float(np.median(vals))
    return {
        "supported_temperatures_celsius": temps,
        "double_sided_median_hcfirst_by_temp": per_temp_median,
    }


def build_profile_fit(tables: CanonicalTables, *, direction_bias_min_strength: float = 1.15) -> dict:
    train_hcf = partition_rd(tables.rd["rd_hcf"]).train
    train_ber = partition_rd(tables.rd["rd_ber"]).train
    train_rp = partition_rd(tables.rd["rd_rp"]).train

    families = sorted({r.family for r in tables.rd["rd_hcf"]})
    fams: dict[str, dict] = {}
    for family in families:
        chips = sorted({r.chip for r in tables.rd["rd_hcf"] if r.family == family})
        strata: dict[str, dict] = {}
        multiplicity: dict[str, list[list[float]]] = {}
        for aggr_class in AGGR_CLASSES:
            for pattern in DATA_PATTERNS:
                key = stratum_key(aggr_class, pattern)
                strata[key] = _fit_stratum(train_hcf, family, aggr_class, pattern)
                multiplicity[key] = _multiplicity(train_ber, family, aggr_class, pattern)
        fams[family] = {
            "chips": chips,
            "chips_train": [c for c in chips if c not in HOLDOUT_CHIPS],
            "chips_holdout": [c for c in chips if c in HOLDOUT_CHIPS],
            "strata": strata,
            "multiplicity": multiplicity,
            "single_to_double_ratio": _single_double_ratio(train_hcf, family),
            "direction": _direction(train_hcf, family, direction_bias_min_strength),
            "rowpress": _rowpress(train_hcf, train_rp, family),
            "temperature": _temperature(tables.ds_ber, family),
        }

    return {
        "method": "lognormal-mle-per-stratum",
        "strata": list(STRATA),
        "sampling_model": (
            "log N_threshold(row) = mu + module_offset + row_eps; "
            "module_offset ~ Normal(0, sigma_between_chip); "
            "row_eps ~ Normal(0, sigma_within_chip)"
        ),
        "families": fams,
    }
