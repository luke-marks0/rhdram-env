"""Distribution fitting helpers (lognormal MLE, quantiles, one-sample KS).

HC-first activation counts across victim rows are right-skewed and positive; a
lognormal is the standard first-order model in the read-disturbance literature and
is supported by the per-stratum log-skew of this data. All routines are pure NumPy
plus :func:`math.erf`, so no external solver is required and results are deterministic.
"""

from __future__ import annotations

import math

import numpy as np

QUANTILES = (1, 5, 25, 50, 75, 95, 99)

_erf = np.vectorize(math.erf, otypes=[float])


def lognormal_fit(values: np.ndarray) -> dict[str, float]:
    """Maximum-likelihood lognormal fit on strictly-positive activation counts."""
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size < 2:
        raise ValueError("need at least two positive samples to fit")
    logs = np.log(values)
    mu = float(logs.mean())
    sigma = float(logs.std(ddof=0))
    return {"mu": mu, "sigma": sigma, "n": int(values.size), "median": float(math.exp(mu))}


def quantiles(values: np.ndarray, qs: tuple[int, ...] = QUANTILES) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    out = {f"p{q:02d}": float(np.percentile(values, q)) for q in qs}
    out["min"] = float(values.min())
    out["max"] = float(values.max())
    return out


def normal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (np.asarray(x, dtype=float) - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + _erf(z))


def ks_statistic(values: np.ndarray, mu: float, sigma: float) -> float:
    """One-sample Kolmogorov-Smirnov D between log(values) and Normal(mu, sigma)."""
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        raise ValueError("no positive samples")
    logs = np.sort(np.log(values))
    n = logs.size
    cdf = normal_cdf(logs, mu, sigma)
    upper = np.arange(1, n + 1) / n - cdf
    lower = cdf - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))
