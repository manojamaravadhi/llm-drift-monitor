from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.engine import Engine

from llmdrift.config import EvalConfig, ModelConfig, load_eval_config, load_models, load_pricing
from llmdrift.datasets.schema import GoldenItem, load_dataset
from llmdrift.storage.db import get_engine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never see real keys, so nothing can hit the network by accident."""
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "OLLAMA_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")


@pytest.fixture(scope="session")
def dataset() -> list[GoldenItem]:
    return load_dataset(ROOT / "datasets" / "golden.jsonl")


@pytest.fixture
def eval_cfg() -> EvalConfig:
    cfg = load_eval_config(ROOT / "config" / "eval.yaml")
    cfg.generation.repeats = 2
    cfg.retry.base_delay_s = 0
    cfg.retry.max_delay_s = 0
    return cfg


@pytest.fixture
def demo_models() -> list[ModelConfig]:
    return load_models(ROOT / "config" / "demo_models.yaml")


@pytest.fixture
def degraded_models() -> list[ModelConfig]:
    return load_models(ROOT / "config" / "demo_models_degraded.yaml")


@pytest.fixture
def pricing() -> dict[str, object]:
    return dict(load_pricing(ROOT / "config" / "pricing.yaml"))


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return get_engine(tmp_path / "test.db")
