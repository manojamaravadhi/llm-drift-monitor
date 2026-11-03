"""Combine metric values into one quality score per response."""

from __future__ import annotations

from collections.abc import Sequence

from llmdrift.metrics.deterministic import MetricValue


def quality_score(metrics: Sequence[MetricValue]) -> float | None:
    """Unweighted mean of all metric values that could be computed (each in [0, 1])."""
    values = [m.value for m in metrics if m.value is not None]
    if not values:
        return None
    return sum(values) / len(values)
