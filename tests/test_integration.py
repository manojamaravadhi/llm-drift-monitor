"""Optional: hits real APIs. Run with `pytest -m integration` and keys in the environment."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from llmdrift.config import load_models
from llmdrift.prompting import build_request
from llmdrift.providers.registry import build_provider, check_available
from tests.conftest import ROOT

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _real_env(monkeypatch: pytest.MonkeyPatch) -> None:
    load_dotenv(ROOT / ".env", override=True)
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "OLLAMA_API_BASE"):
        if os.environ.get(var):
            monkeypatch.setenv(var, os.environ[var])


@pytest.mark.parametrize(
    "model",
    [m for m in load_models(ROOT / "config" / "models.yaml") if m.enabled],
    ids=lambda m: m.name,
)
def test_real_provider_answers(model, dataset) -> None:  # type: ignore[no-untyped-def]
    ok, reason = check_available(model)
    if not ok:
        pytest.skip(reason)
    item = next(i for i in dataset if i.id == "fact-003")
    result = build_provider(model, dataset).generate(build_request(item, "Be brief.", 0.0, 64, 0))
    assert "canberra" in result.text.lower()
    assert result.latency_ms > 0 and result.output_tokens
