from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from llmdrift.analysis import load_results
from llmdrift.config import DriftConfig, EvalConfig, ModelConfig
from llmdrift.datasets.schema import GoldenItem
from llmdrift.drift.baseline import BaselineError, choose_baseline
from llmdrift.drift.detector import detect_drift, to_markdown
from llmdrift.providers.retry import RetryPolicy, Throttle
from llmdrift.runner import Runner, RunOptions
from llmdrift.storage.db import set_baseline

CFG = DriftConfig(bootstrap_samples=1000)


def _frame(
    run: str,
    scores: list[float],
    model: str = "m",
    latency: float = 500.0,
    cost: float = 1e-5,
    error_every: int = 0,
    refused: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, s in enumerate(scores):
        err = "timeout" if error_every and i % error_every == 0 else None
        rows.append(
            {
                "run_id": run,
                "model_name": model,
                "prompt_id": f"p{i:03d}",
                "category": "cat_a" if i % 2 else "cat_b",
                "expected_behavior": "answer",
                "repeat_idx": 0,
                "error_type": err,
                "quality_score": None if err else s,
                "latency_ms": latency * rng.uniform(0.9, 1.1),
                "cost_list_usd": cost,
                "refused": refused,
            }
        )
    return pd.DataFrame(rows)


def _flags(report: Any) -> set[tuple[str, str]]:
    return {(f.category, f.metric) for f in report.flagged}


def test_no_drift_on_noise() -> None:
    rng = np.random.default_rng(42)
    base_scores = rng.uniform(0.5, 1.0, 40)
    noisy = np.clip(base_scores + rng.normal(0, 0.05, 40), 0, 1)
    report = detect_drift(
        _frame("cur", list(noisy), seed=1),
        _frame("base", list(base_scores), seed=2),
        CFG,
        "cur",
        ["base"],
        "explicit",
    )
    assert not report.drift_detected, to_markdown(report)


def test_small_noisy_drop_below_threshold_not_flagged() -> None:
    base = [0.8] * 40
    cur = [0.75] * 40  # consistent but small (-0.05 < threshold 0.10)
    report = detect_drift(_frame("c", cur), _frame("b", base), CFG, "c", ["b"], "explicit")
    assert ("ALL", "quality") not in _flags(report)


def test_large_but_inconsistent_drop_not_flagged() -> None:
    """Mean drops by > threshold, but driven by 2 prompts: CI includes 0 -> no alarm."""
    base = [0.9] * 6
    cur = [0.9, 0.9, 0.9, 0.9, 0.0, 0.3]
    cfg = CFG.model_copy(update={"min_prompts": 5})
    report = detect_drift(_frame("c", cur), _frame("b", base), cfg, "c", ["b"], "explicit")
    finding = next(f for f in report.findings if f.category == "ALL" and f.metric == "quality")
    assert finding.delta is not None and finding.delta <= -0.1
    assert finding.status == "ok"


def test_clear_quality_drift_flagged() -> None:
    rng = np.random.default_rng(7)
    base = rng.uniform(0.8, 1.0, 40)
    report = detect_drift(
        _frame("c", list(base - 0.3)), _frame("b", list(base)), CFG, "c", ["b"], "explicit"
    )
    assert {("ALL", "quality"), ("cat_a", "quality"), ("cat_b", "quality")} <= _flags(report)
    assert report.to_dict()["drift_detected"] is True
    assert "DRIFT DETECTED" in to_markdown(report)


def test_latency_cost_error_and_refusal_regressions() -> None:
    base = _frame("b", [0.9] * 40)
    cur = _frame("c", [0.9] * 40, latency=1000, cost=2e-5, error_every=4, refused=True)
    flags = _flags(detect_drift(cur, base, CFG, "c", ["b"], "explicit"))
    assert {
        ("ALL", "latency_p95_ms"),
        ("ALL", "cost_list_per_query_usd"),
        ("ALL", "error_rate"),
        ("ALL", "over_refusal_rate"),
    } <= flags
    assert ("ALL", "quality") not in flags


def test_insufficient_data_never_flagged() -> None:
    report = detect_drift(
        _frame("c", [0.0] * 3), _frame("b", [1.0] * 3), CFG, "c", ["b"], "explicit"
    )
    assert not report.drift_detected
    assert {f.status for f in report.findings} == {"insufficient_data"}


def test_rolling_baseline_uses_median_per_prompt() -> None:
    b1, b2, b3 = _frame("b1", [0.9] * 20), _frame("b2", [0.9] * 20), _frame("b3", [0.0] * 20)
    baseline = pd.concat([b1, b2, b3])
    report = detect_drift(_frame("c", [0.9] * 20), baseline, CFG, "c", ["b1", "b2", "b3"], "r")
    q = next(f for f in report.findings if f.category == "ALL" and f.metric == "quality")
    assert q.baseline == pytest.approx(0.9)  # outlier run b3 ignored by the median


def _seed_runs(
    eval_cfg: EvalConfig,
    models_seq: list[list[ModelConfig]],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> list[str]:
    ids = []
    for models in models_seq:
        runner = Runner(
            eval_cfg,
            models,
            pricing,
            dataset,
            engine,
            retry_policy=RetryPolicy(sleep=lambda _: None),
            throttle_factory=lambda rpm: Throttle(rpm=1e9, sleep=lambda _: None),
        )
        ids.append(runner.run(RunOptions(fresh=True, use_cache=False)).run_id)
    return ids


def test_end_to_end_injected_degradation(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    degraded_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    ids = _seed_runs(
        eval_cfg, [demo_models, demo_models, degraded_models], pricing, dataset, engine
    )
    choice = choose_baseline(engine, eval_cfg.drift)
    assert choice.current_run == ids[-1] and choice.baseline_runs == ids[:2]
    report = detect_drift(
        load_results(engine, [choice.current_run]),
        load_results(engine, choice.baseline_runs),
        eval_cfg.drift,
        choice.current_run,
        choice.baseline_runs,
        choice.mode,
    )
    flagged_models = {f.model for f in report.flagged}
    assert flagged_models == {"mock-good"}

    # Healthy re-run of the same config: no drift.
    healthy = choose_baseline(engine, eval_cfg.drift, current_run=ids[1], baseline_run=ids[0])
    report2 = detect_drift(
        load_results(engine, [ids[1]]),
        load_results(engine, [ids[0]]),
        eval_cfg.drift,
        ids[1],
        [ids[0]],
        healthy.mode,
    )
    assert not report2.drift_detected


def test_baseline_selection_rules(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    with pytest.raises(BaselineError, match="no completed"):
        choose_baseline(engine, eval_cfg.drift)
    small = [m.model_copy() for m in demo_models[:1]]
    eval_cfg.generation.repeats = 1
    ids = _seed_runs(eval_cfg, [small] * 4, pricing, dataset, engine)
    with pytest.raises(BaselineError, match="earlier"):
        choose_baseline(engine, eval_cfg.drift, current_run=ids[0])
    cfg = eval_cfg.drift.model_copy(update={"rolling_k": 2})
    assert choose_baseline(engine, cfg).baseline_runs == ids[1:3]
    assert choose_baseline(engine, cfg, baseline_run=ids[0]).mode == "explicit"
    with pytest.raises(BaselineError, match="same"):
        choose_baseline(engine, cfg, baseline_run=ids[-1])
    with pytest.raises(BaselineError, match="not found"):
        choose_baseline(engine, cfg, baseline_run="nope")
    with pytest.raises(BaselineError, match="not found"):
        choose_baseline(engine, cfg, current_run="nope")
    with Session(engine) as s:
        set_baseline(s, ids[0])
        s.commit()
        with pytest.raises(KeyError):
            set_baseline(s, "nope")
    designated = choose_baseline(engine, cfg)
    assert designated.mode == "designated" and designated.baseline_runs == [ids[0]]
    named = cfg.model_copy(update={"baseline": ids[1]})
    assert choose_baseline(engine, named).baseline_runs == [ids[1]]
