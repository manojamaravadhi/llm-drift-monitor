from __future__ import annotations

from typing import Any

import pytest

from llmdrift.config import EvalConfig, ModelConfig
from llmdrift.live import adhoc_item, ask_models
from llmdrift.providers.base import RateLimited


def test_ask_models_in_order_with_skips(
    eval_cfg: EvalConfig, mock_models: list[ModelConfig], pricing: dict[str, Any]
) -> None:
    disabled = ModelConfig(name="off", provider="groq", model="groq/x", enabled=False)
    answers = ask_models(
        [*mock_models, disabled], "What is the chemical symbol for gold?",
        eval_cfg.generation, pricing,
    )  # fmt: skip
    assert [a.model_name for a in answers] == ["mock-good", "mock-fast", "off"]
    assert answers[2].skipped == "disabled in config"
    assert answers[0].text and answers[0].latency_ms and answers[0].list_usd is not None
    assert ask_models([], "hello there", eval_cfg.generation, pricing) == []


def test_provider_errors_are_reported_not_raised(
    eval_cfg: EvalConfig,
    mock_models: list[ModelConfig],
    pricing: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Down:
        def generate(self, request: Any) -> Any:
            raise RateLimited("429 slow down")

    monkeypatch.setattr("llmdrift.live.build_provider", lambda cfg, items: Down())
    answers = ask_models(mock_models[:1], "Hi", eval_cfg.generation, pricing)
    assert answers[0].error == "rate_limit: 429 slow down"


def test_adhoc_item_uses_rag_template_with_context() -> None:
    item = adhoc_item("Who?", "Ada founded it.")
    assert item.context == "Ada founded it." and item.category.value == "rag"
    assert adhoc_item("Who?").context is None
