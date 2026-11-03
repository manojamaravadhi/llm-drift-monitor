"""Runs the golden dataset against every model and stores the results."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from llmdrift.config import EvalConfig, ModelConfig, PriceEntry, stable_hash
from llmdrift.datasets.schema import GoldenItem
from llmdrift.metrics.cost import compute_cost
from llmdrift.metrics.deterministic import MetricValue, is_refusal, score_deterministic
from llmdrift.metrics.judge import Judge, NullJudge
from llmdrift.metrics.scoring import quality_score
from llmdrift.prompting import build_request, render_prompt
from llmdrift.providers.base import CompletionResult, GenerationRequest, Provider, ProviderError
from llmdrift.providers.registry import build_provider, check_available
from llmdrift.providers.retry import CircuitBreaker, RetryPolicy, Throttle
from llmdrift.storage.db import (
    MetricRow,
    Result,
    Run,
    get_cached,
    latest_resumable_run,
    put_cached,
    utcnow,
)

log = logging.getLogger(__name__)

ProviderFactory = Callable[[ModelConfig, list[GoldenItem]], Provider]


class NoModelsAvailable(RuntimeError):
    """Raised when every selected model is disabled or unreachable."""


@dataclass
class RunOptions:
    """Per-invocation options (mostly from the CLI)."""

    models: list[str] | None = None
    categories: list[str] | None = None
    repeats: int | None = None
    use_cache: bool = True
    fresh: bool = False
    inject_degradation: bool = False
    label: str | None = None
    synthetic: bool = False
    started_at: datetime | None = None  # override for seeding synthetic history
    run_id: str | None = None


@dataclass
class RunSummary:
    """What happened in a run."""

    run_id: str
    resumed: bool
    models_run: list[str]
    models_skipped: dict[str, str]
    calls_made: int = 0
    cache_hits: int = 0
    skipped_existing: int = 0
    errors: int = 0
    per_model_errors: dict[str, int] = field(default_factory=dict)


def git_commit() -> str | None:
    """Current commit hash (GITHUB_SHA in CI), or ``None`` outside a git checkout."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def dataset_hash(items: list[GoldenItem]) -> str:
    """Hash of the dataset content, stored with each run."""
    return stable_hash([i.model_dump(mode="json") for i in items])


def cache_key(model_cfg: ModelConfig, request: GenerationRequest) -> str:
    """Response cache key: same model + messages + params + repeat -> same response."""
    return stable_hash(
        {
            "model": model_cfg.model,
            "provider_params": model_cfg.params,
            "messages": request.as_litellm_messages(),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "repeat": request.repeat_idx,
        }
    )


class Runner:
    """Evaluate models on the golden dataset and persist every result."""

    def __init__(
        self,
        eval_cfg: EvalConfig,
        models: list[ModelConfig],
        pricing: dict[str, PriceEntry],
        dataset: list[GoldenItem],
        engine: Engine,
        judge: Judge | None = None,
        provider_factory: ProviderFactory = build_provider,
        availability: Callable[[ModelConfig], tuple[bool, str]] = check_available,
        retry_policy: RetryPolicy | None = None,
        throttle_factory: Callable[[float], Throttle] = lambda rpm: Throttle(rpm=rpm),
    ) -> None:
        """Wire dependencies. Everything with side effects is injectable for tests."""
        self.cfg = eval_cfg
        self.models = models
        self.pricing = pricing
        self.dataset = dataset
        self.engine = engine
        self.judge = judge or NullJudge()
        self.provider_factory = provider_factory
        self.availability = availability
        self.retry = retry_policy or RetryPolicy(
            max_attempts=eval_cfg.retry.max_attempts,
            base_delay_s=eval_cfg.retry.base_delay_s,
            max_delay_s=eval_cfg.retry.max_delay_s,
        )
        self.throttle_factory = throttle_factory

    # ----------------------------------------------------------------- selection
    def _select_items(self, opts: RunOptions) -> list[GoldenItem]:
        if not opts.categories:
            return list(self.dataset)
        wanted = set(opts.categories)
        return [i for i in self.dataset if i.category.value in wanted]

    def _select_models(self, opts: RunOptions) -> tuple[list[ModelConfig], dict[str, str]]:
        chosen, skipped = [], {}
        known = {m.name for m in self.models}
        for name in opts.models or []:
            if name not in known:
                skipped[name] = "not defined in models config"
        for m in self.models:
            if opts.models and m.name not in opts.models:
                continue
            ok, reason = self.availability(m)
            if ok:
                chosen.append(m)
            else:
                skipped[m.name] = reason
        return chosen, skipped

    def _snapshot(
        self, models: list[ModelConfig], items: list[GoldenItem], opts: RunOptions, repeats: int
    ) -> dict[str, object]:
        gen = self.cfg.generation.model_copy(update={"repeats": repeats})
        return {
            "models": [m.model_dump(mode="json") for m in models],
            "generation": gen.model_dump(mode="json"),
            "judge": self.cfg.judge.model_dump(mode="json")
            if not isinstance(self.judge, NullJudge)
            else {"enabled": False},
            "dataset_hash": dataset_hash(items),
            "n_items": len(items),
            "categories": sorted({i.category.value for i in items}),
            "inject_degradation": opts.inject_degradation,
            "use_cache": opts.use_cache,
        }

    # ----------------------------------------------------------------- main loop
    def run(self, opts: RunOptions | None = None) -> RunSummary:
        """Run (or resume) an evaluation. Returns a summary; results are in the DB."""
        opts = opts or RunOptions()
        items = self._select_items(opts)
        if not items:
            raise ValueError("no dataset items match the selected categories")
        models, skipped = self._select_models(opts)
        for name, reason in skipped.items():
            log.warning("skipping model %s: %s", name, reason)
        if not models:
            raise NoModelsAvailable(f"no usable models ({skipped})")
        repeats = opts.repeats or self.cfg.generation.repeats

        snapshot = self._snapshot(models, items, opts, repeats)
        config_hash = stable_hash(snapshot)
        with Session(self.engine, expire_on_commit=False) as session:
            run = None if opts.fresh else latest_resumable_run(session, config_hash)
            resumed = run is not None
            if run is None:
                run = Run(
                    run_id=opts.run_id or f"{utcnow():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
                    started_at=opts.started_at or utcnow(),
                    git_commit=git_commit(),
                    config_json=json.dumps(snapshot, sort_keys=True),
                    config_hash=config_hash,
                    dataset_hash=str(snapshot["dataset_hash"]),
                    label=opts.label,
                    synthetic=opts.synthetic,
                )
                session.add(run)
                session.commit()
            else:
                log.info("resuming run %s", run.run_id)

            summary = RunSummary(
                run_id=run.run_id,
                resumed=resumed,
                models_run=[m.name for m in models],
                models_skipped=skipped,
            )
            for model_cfg in models:
                self._run_model(session, run, model_cfg, items, repeats, opts, summary)

            run.status = "completed"
            run.finished_at = utcnow() if opts.started_at is None else opts.started_at
            session.commit()
        return summary

    def _run_model(
        self,
        session: Session,
        run: Run,
        model_cfg: ModelConfig,
        items: list[GoldenItem],
        repeats: int,
        opts: RunOptions,
        summary: RunSummary,
    ) -> None:
        provider = self.provider_factory(model_cfg, self.dataset)
        throttle = self.throttle_factory(model_cfg.rpm)
        breaker = CircuitBreaker(threshold=self.cfg.retry.circuit_breaker_failures)
        existing = {
            (r.prompt_id, r.repeat_idx): r
            for r in session.scalars(
                select(Result).where(
                    Result.run_id == run.run_id, Result.model_name == model_cfg.name
                )
            )
        }
        gen = self.cfg.generation
        log.info("model %s: %d items x %d repeats", model_cfg.name, len(items), repeats)
        for item in items:
            for repeat_idx in range(repeats):
                prior = existing.get((item.id, repeat_idx))
                if prior is not None and prior.error_type is None:
                    summary.skipped_existing += 1
                    continue
                if prior is not None:
                    session.delete(prior)  # retry previously failed rows
                    session.flush()
                request = build_request(
                    item,
                    gen.system_prompt,
                    gen.temperature,
                    gen.max_tokens,
                    repeat_idx,
                    opts.inject_degradation,
                )
                row = self._evaluate_one(
                    provider, model_cfg, item, request, throttle, breaker, opts, session, summary
                )
                row.run_id = run.run_id
                if opts.started_at is not None:
                    row.created_at = opts.started_at
                session.add(row)
                session.commit()

    def _evaluate_one(
        self,
        provider: Provider,
        model_cfg: ModelConfig,
        item: GoldenItem,
        request: GenerationRequest,
        throttle: Throttle,
        breaker: CircuitBreaker,
        opts: RunOptions,
        session: Session,
        summary: RunSummary,
    ) -> Result:
        key = cache_key(model_cfg, request)
        row = Result(
            model_name=model_cfg.name,
            model_id=model_cfg.model,
            prompt_id=item.id,
            category=item.category.value,
            expected_behavior=item.expected_behavior.value,
            repeat_idx=request.repeat_idx,
            cache_key=key,
        )
        completion: CompletionResult | None = None
        cached = get_cached(session, key) if opts.use_cache else None
        if cached is not None:
            completion = CompletionResult(**cached)
            row.from_cache = True
            summary.cache_hits += 1
        elif breaker.is_open:
            row.error_type = "circuit_open"
            row.error_message = "skipped after repeated provider failures"
        else:
            try:
                throttle.wait()
                summary.calls_made += 1
                fresh, row.attempts = self.retry.call(lambda: provider.generate(request))
                breaker.record(success=True)
                put_cached(session, key, model_cfg.model, asdict(fresh))
                completion = fresh
            except ProviderError as exc:
                breaker.record(success=False)
                row.error_type = exc.kind
                row.error_message = str(exc)[:500]
                row.attempts = self.retry.max_attempts if exc.retryable else 1

        if completion is None:
            summary.errors += 1
            summary.per_model_errors[model_cfg.name] = (
                summary.per_model_errors.get(model_cfg.name, 0) + 1
            )
            return row

        row.response_text = completion.text
        row.latency_ms = completion.latency_ms
        row.ttft_ms = completion.ttft_ms
        row.input_tokens = completion.input_tokens
        row.output_tokens = completion.output_tokens
        row.tokens_estimated = completion.tokens_estimated
        cost = compute_cost(
            completion.input_tokens, completion.output_tokens, self.pricing.get(model_cfg.model)
        )
        row.cost_billed_usd = cost.billed
        row.cost_list_usd = cost.list_price
        row.refused = is_refusal(completion.text)

        metrics: list[MetricValue] = score_deterministic(item, completion.text)
        if request.repeat_idx < self.cfg.judge.max_repeats_judged:
            metrics.extend(self.judge.score(item, render_prompt(item), completion.text))
        row.quality_score = quality_score(metrics)
        row.metrics = [
            MetricRow(name=m.name, value=m.value, source=m.source, reason=m.reason or None)
            for m in metrics
        ]
        return row
