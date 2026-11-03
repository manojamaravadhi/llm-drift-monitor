from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from llmdrift.config import EvalConfig, ModelConfig
from llmdrift.datasets.schema import GoldenItem
from llmdrift.metrics.deterministic import MetricValue
from llmdrift.providers.base import (
    CompletionResult,
    GenerationRequest,
    Provider,
    ProviderAuthError,
    ProviderUnavailable,
)
from llmdrift.providers.registry import build_provider
from llmdrift.providers.retry import RetryPolicy, Throttle
from llmdrift.runner import NoModelsAvailable, Runner, RunOptions, git_commit
from llmdrift.storage.db import MetricRow, Result, Run


def _no_wait(_rpm: float) -> Throttle:
    return Throttle(rpm=1e9, sleep=lambda _: None)


def _runner(
    eval_cfg: EvalConfig,
    models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
    **kw: Any,
) -> Runner:
    kw.setdefault("throttle_factory", _no_wait)
    kw.setdefault("retry_policy", RetryPolicy(max_attempts=2, sleep=lambda _: None))
    return Runner(eval_cfg, models, pricing, dataset, engine, **kw)


def _count(engine: Engine, model: type = Result) -> int:
    with Session(engine) as s:
        return int(s.scalar(select(func.count()).select_from(model)) or 0)


def test_full_mock_run_stores_everything(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    summary = _runner(eval_cfg, demo_models, pricing, dataset, engine).run()
    expected = len(dataset) * 2 * len(demo_models)
    assert summary.calls_made == expected
    assert _count(engine) == expected
    assert _count(engine, MetricRow) > expected
    with Session(engine) as s:
        run = s.get(Run, summary.run_id)
        assert run is not None and run.status == "completed"
        assert run.config["n_items"] == len(dataset)
        row = s.scalars(select(Result).where(Result.error_type.is_(None))).first()
        assert row is not None
        assert row.quality_score is not None and row.latency_ms and row.cost_list_usd
        assert row.cost_billed_usd == 0.0


def test_cache_prevents_repaying_and_can_be_bypassed(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    runner = _runner(eval_cfg, demo_models, pricing, dataset, engine)
    runner.run(RunOptions(categories=["factual"]))
    second = runner.run(RunOptions(categories=["factual"], fresh=True))
    assert second.calls_made == 0 and second.cache_hits == 8 * 2 * 2
    third = runner.run(RunOptions(categories=["factual"], fresh=True, use_cache=False))
    assert third.calls_made == 8 * 2 * 2


class Interrupting:
    """Wraps a provider and raises KeyboardInterrupt after N calls (simulated crash)."""

    def __init__(self, inner: Provider, after: int) -> None:
        self.inner, self.after, self.calls = inner, after, 0
        self.name, self.model_id = inner.name, inner.model_id

    def generate(self, request: GenerationRequest) -> CompletionResult:
        self.calls += 1
        if self.calls > self.after:
            raise KeyboardInterrupt
        return self.inner.generate(request)


def test_interrupted_run_resumes_without_duplicates(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    models = demo_models[:1]
    opts = RunOptions(categories=["factual", "reasoning"], use_cache=False)
    crashing = _runner(
        eval_cfg,
        models,
        pricing,
        dataset,
        engine,
        provider_factory=lambda cfg, ds: Interrupting(build_provider(cfg, ds), after=10),
    )
    with pytest.raises(KeyboardInterrupt):
        crashing.run(opts)
    assert _count(engine) == 10

    summary = _runner(eval_cfg, models, pricing, dataset, engine).run(opts)
    assert summary.resumed
    assert summary.skipped_existing == 10
    assert summary.calls_made == 16 * 2 - 10
    assert _count(engine) == 16 * 2
    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(Run)) == 1


class Broken:
    name, model_id = "broken", "mock/broken"

    def __init__(self, exc: Exception) -> None:
        self.exc, self.calls = exc, 0

    def generate(self, request: GenerationRequest) -> CompletionResult:
        self.calls += 1
        raise self.exc


def test_failing_provider_is_recorded_and_circuit_breaks(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    broken = Broken(ProviderUnavailable("outage"))
    eval_cfg.retry.circuit_breaker_failures = 3
    runner = _runner(
        eval_cfg,
        demo_models[:1],
        pricing,
        dataset,
        engine,
        provider_factory=lambda cfg, ds: broken,
    )
    summary = runner.run(RunOptions(categories=["factual"]))
    assert summary.errors == 16
    assert broken.calls == 3 * 2  # 3 failed requests x 2 attempts, then circuit opens
    with Session(engine) as s:
        kinds = set(s.scalars(select(Result.error_type)))
    assert kinds == {"unavailable", "circuit_open"}

    # Simulate an interrupted run: on resume, failed rows are retried (now with a working
    # provider) and successful rows are kept.
    with Session(engine) as s:
        run = s.get(Run, summary.run_id)
        assert run is not None
        run.status = "running"
        s.commit()
    summary2 = _runner(eval_cfg, demo_models[:1], pricing, dataset, engine).run(
        RunOptions(categories=["factual"])
    )
    assert summary2.resumed and summary2.calls_made == 16 and summary2.errors == 0
    assert _count(engine) == 16


def test_non_retryable_error_single_attempt(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    broken = Broken(ProviderAuthError("bad key"))
    runner = _runner(
        eval_cfg,
        demo_models[:1],
        pricing,
        dataset,
        engine,
        provider_factory=lambda cfg, ds: broken,
    )
    runner.run(RunOptions(categories=["factual"], repeats=1))
    assert broken.calls == 5  # circuit breaker threshold from eval.yaml


def test_model_selection_and_skips(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    def availability(cfg: ModelConfig) -> tuple[bool, str]:
        return (cfg.name == "mock-good", "offline")

    runner = _runner(eval_cfg, demo_models, pricing, dataset, engine, availability=availability)
    summary = runner.run(
        RunOptions(models=["mock-good", "mock-fast", "ghost"], repeats=1, categories=["safety"])
    )
    assert summary.models_run == ["mock-good"]
    assert summary.models_skipped == {
        "ghost": "not defined in models config",
        "mock-fast": "offline",
    }
    with pytest.raises(NoModelsAvailable):
        runner.run(RunOptions(models=["mock-fast"]))
    with pytest.raises(ValueError, match="categories"):
        runner.run(RunOptions(categories=["nope"]))


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, item: GoldenItem, prompt: str, response: str) -> list[MetricValue]:
        self.calls += 1
        return [MetricValue("judge_correctness", 0.5, "judge", "fake")]


def test_judge_only_scores_first_repeats(
    eval_cfg: EvalConfig,
    demo_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
    engine: Engine,
) -> None:
    judge = FakeJudge()
    eval_cfg.judge.max_repeats_judged = 1
    runner = _runner(eval_cfg, demo_models[:1], pricing, dataset, engine, judge=judge)
    summary = runner.run(RunOptions(categories=["factual"], repeats=3))
    assert judge.calls == 8
    with Session(engine) as s:
        run = s.get(Run, summary.run_id)
        assert run is not None and run.config["judge"]["model"] == eval_cfg.judge.model


def test_git_commit_prefers_ci_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567890")
    assert git_commit() == "abcdef123456"
