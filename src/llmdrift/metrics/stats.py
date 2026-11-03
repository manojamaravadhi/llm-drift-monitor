"""Small statistics helpers: percentiles and bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in 0..100); ``None`` for empty input."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(np.percentile(np.asarray(clean, dtype=float), q))


def latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    """p50, p95 and mean of a latency sample (milliseconds)."""
    clean = [v for v in values if v is not None]
    return {
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "mean": float(np.mean(clean)) if clean else None,
    }


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate of ``stat(current) - stat(baseline)`` and its confidence interval."""

    delta: float
    ci_low: float
    ci_high: float
    n_current: int
    n_baseline: int


def bootstrap_paired_mean_diff(
    current: Sequence[float],
    baseline: Sequence[float],
    n_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap of ``mean(current - baseline)``. Inputs are aligned per prompt."""
    cur = np.asarray(current, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if cur.shape != base.shape or cur.size == 0:
        raise ValueError("paired samples must be non-empty and the same length")
    diffs = cur - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_samples, diffs.size))
    means = diffs[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return BootstrapResult(
        delta=float(diffs.mean()),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1 - alpha)),
        n_current=int(cur.size),
        n_baseline=int(base.size),
    )


def bootstrap_stat_diff(
    current: Sequence[float],
    baseline: Sequence[float],
    stat: Callable[[FloatArray], float],
    n_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Unpaired bootstrap of ``stat(current) - stat(baseline)`` (e.g. p95 latency, a rate)."""
    cur = np.asarray(current, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if cur.size == 0 or base.size == 0:
        raise ValueError("samples must be non-empty")
    rng = np.random.default_rng(seed)
    cur_idx = rng.integers(0, cur.size, size=(n_samples, cur.size))
    base_idx = rng.integers(0, base.size, size=(n_samples, base.size))
    diffs = np.array(
        [stat(cur[ci]) - stat(base[bi]) for ci, bi in zip(cur_idx, base_idx, strict=True)]
    )
    alpha = (1 - confidence) / 2
    return BootstrapResult(
        delta=float(stat(cur) - stat(base)),
        ci_low=float(np.quantile(diffs, alpha)),
        ci_high=float(np.quantile(diffs, 1 - alpha)),
        n_current=int(cur.size),
        n_baseline=int(base.size),
    )


def p95(values: FloatArray) -> float:
    """95th percentile of an array."""
    return float(np.percentile(values, 95))


def mean(values: FloatArray) -> float:
    """Arithmetic mean of an array."""
    return float(np.mean(values))
