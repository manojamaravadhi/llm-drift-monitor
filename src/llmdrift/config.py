"""Typed loading of the YAML configuration files."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ProviderKind = Literal["gemini", "groq", "ollama", "openai_compatible", "mock"]


class ModelConfig(BaseModel):
    """One model under test (an entry of ``models.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: ProviderKind
    model: str
    enabled: bool = True
    api_key_env: str | None = None
    api_base: str | None = None
    rpm: float = Field(default=30, gt=0)
    timeout_s: float = Field(default=60, gt=0)
    stream: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsFile(BaseModel):
    """Top level of ``models.yaml``."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelConfig]

    @field_validator("models")
    @classmethod
    def _unique_names(cls, models: list[ModelConfig]) -> list[ModelConfig]:
        names = [m.name for m in models]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate model names: {sorted(dupes)}")
        return models


class PriceEntry(BaseModel):
    """List price for one model, per 1M tokens (USD)."""

    model_config = ConfigDict(extra="forbid")

    input_per_1m: float | None = Field(default=None, ge=0)
    output_per_1m: float | None = Field(default=None, ge=0)
    free_tier: bool = False
    source_url: str
    checked_on: date


class PricingFile(BaseModel):
    """Top level of ``pricing.yaml``."""

    model_config = ConfigDict(extra="forbid")

    prices: dict[str, PriceEntry]


class GenerationConfig(BaseModel):
    """Sampling settings shared by every model."""

    model_config = ConfigDict(extra="forbid")

    repeats: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0)
    system_prompt: str = "You are a helpful, accurate assistant. Answer concisely."


class RetryConfig(BaseModel):
    """Retry and circuit-breaker settings."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=5, ge=1)
    base_delay_s: float = Field(default=1.0, ge=0)
    max_delay_s: float = Field(default=30.0, ge=0)
    circuit_breaker_failures: int = Field(default=5, ge=1)


class JudgeConfig(BaseModel):
    """LLM-as-judge settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: str = "gemini/gemini-3.1-flash-lite"
    api_key_env: str | None = "GEMINI_API_KEY"
    api_base: str | None = None
    temperature: float = 0.0
    metrics: list[Literal["correctness", "relevancy", "faithfulness"]] = Field(
        default_factory=lambda: ["correctness", "faithfulness"]  # type: ignore[arg-type]
    )
    max_repeats_judged: int = Field(default=1, ge=1)
    rpm: float = Field(default=10, gt=0)


class DriftConfig(BaseModel):
    """Drift detection thresholds."""

    model_config = ConfigDict(extra="forbid")

    baseline: str = "rolling"
    rolling_k: int = Field(default=3, ge=1)
    # How baseline runs are combined per prompt. "mean" is unbiased; "median" resists one bad
    # baseline run but rounds bounded, mostly-perfect scores upward (more false alarms).
    rolling_stat: Literal["mean", "median"] = "mean"
    quality_threshold: float = Field(default=0.10, ge=0)
    # Per-category groups have ~8 prompts, so they get a larger threshold and a
    # Bonferroni-adjusted confidence level (many categories are tested at once).
    category_quality_threshold: float = Field(default=0.20, ge=0)
    bonferroni: bool = True
    latency_threshold: float = Field(default=0.50, ge=0)
    cost_threshold: float = Field(default=0.25, ge=0)
    rate_threshold: float = Field(default=0.10, ge=0)
    min_prompts: int = Field(default=5, ge=1)
    bootstrap_samples: int = Field(default=2000, ge=100)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    seed: int = 1234


class EvalConfig(BaseModel):
    """Top level of ``eval.yaml``."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = "datasets/golden.jsonl"
    db_path: str = "data/results.db"
    reports_dir: str = "reports"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_models(path: str | Path) -> list[ModelConfig]:
    """Load all model entries (enabled or not) from ``models.yaml``."""
    return ModelsFile.model_validate(_read_yaml(Path(path))).models


def load_pricing(path: str | Path) -> dict[str, PriceEntry]:
    """Load price entries keyed by LiteLLM model id."""
    return PricingFile.model_validate(_read_yaml(Path(path))).prices


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load ``eval.yaml``."""
    return EvalConfig.model_validate(_read_yaml(Path(path)))


def stable_hash(obj: Any) -> str:
    """SHA-256 of a JSON-serialisable object with sorted keys."""
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
