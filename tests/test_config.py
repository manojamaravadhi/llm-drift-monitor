from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llmdrift.config import load_eval_config, load_models, load_pricing, stable_hash
from tests.conftest import ROOT


def test_shipped_configs_parse() -> None:
    models = load_models(ROOT / "config" / "models.yaml")
    providers = {m.provider for m in models}
    assert {"gemini", "groq", "ollama"} <= providers
    assert all(m.api_key_env or m.provider in {"ollama", "mock"} for m in models)
    load_models(ROOT / "config" / "demo_models.yaml")
    cfg = load_eval_config(ROOT / "config" / "eval.yaml")
    assert cfg.judge.temperature == 0.0
    assert cfg.generation.repeats >= 1


def test_every_enabled_model_has_a_price_entry() -> None:
    prices = load_pricing(ROOT / "config" / "pricing.yaml")
    for path in ("models.yaml", "demo_models.yaml"):
        for m in load_models(ROOT / "config" / path):
            if m.enabled:
                assert m.model in prices, m.model
    for entry in prices.values():
        assert entry.source_url and entry.checked_on


def test_duplicate_model_names_rejected(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(
        "models:\n"
        "  - {name: a, provider: mock, model: mock/a}\n"
        "  - {name: a, provider: mock, model: mock/b}\n"
    )
    with pytest.raises(ValidationError, match="duplicate"):
        load_models(p)


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text("models:\n  - {name: a, provider: mock, model: m, api_key: secret}\n")
    with pytest.raises(ValidationError):
        load_models(p)


def test_bad_provider_rejected(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text("models:\n  - {name: a, provider: openai, model: m}\n")
    with pytest.raises(ValidationError):
        load_models(p)


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text("- just a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_eval_config(p)


def test_stable_hash_is_order_independent() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
