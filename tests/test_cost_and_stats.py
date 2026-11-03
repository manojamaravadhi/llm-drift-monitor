from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from llmdrift.config import PriceEntry
from llmdrift.metrics.cost import compute_cost
from llmdrift.metrics.stats import (
    bootstrap_paired_mean_diff,
    bootstrap_stat_diff,
    latency_summary,
    p95,
    percentile,
)

PRICE = PriceEntry(
    input_per_1m=0.30, output_per_1m=2.50, free_tier=True, source_url="x", checked_on=date.today()
)


def test_cost_free_tier_bills_zero_but_has_list_price() -> None:
    cost = compute_cost(1_000_000, 1_000_000, PRICE)
    assert cost.billed == 0.0
    assert cost.list_price == pytest.approx(2.80)


def test_cost_paid_tier() -> None:
    cost = compute_cost(2000, 500, PRICE, using_free_tier=False)
    assert cost.billed == pytest.approx((2000 * 0.30 + 500 * 2.50) / 1e6)
    assert cost.billed == cost.list_price


def test_cost_unknowns() -> None:
    assert compute_cost(10, 10, None).list_price is None
    assert compute_cost(None, 10, PRICE).list_price is None
    local = PRICE.model_copy(update={"input_per_1m": None})
    assert compute_cost(10, 10, local).list_price is None


def test_percentiles() -> None:
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile([], 50) is None
    summary = latency_summary([100.0, 200.0, 300.0])
    assert summary["p50"] == 200.0
    assert summary["mean"] == 200.0
    assert latency_summary([])["p95"] is None


def test_paired_bootstrap_detects_shift_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 1.0, 40)
    res = bootstrap_paired_mean_diff(base - 0.3, base, seed=1)
    assert res.delta == pytest.approx(-0.3)
    assert res.ci_high < 0
    again = bootstrap_paired_mean_diff(base - 0.3, base, seed=1)
    assert again == res


def test_paired_bootstrap_validates_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_paired_mean_diff([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        bootstrap_stat_diff([], [1.0], p95)


def test_unpaired_bootstrap_ci_contains_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(500, 50, 100), rng.normal(500, 50, 100)
    res = bootstrap_stat_diff(a, b, p95, seed=2)
    assert res.ci_low < 0 < res.ci_high
