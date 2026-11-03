from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from llmdrift.config import ModelConfig
from llmdrift.datasets.schema import GoldenItem
from llmdrift.prompting import DEGRADED_SYSTEM_PROMPT, build_request, render_prompt
from llmdrift.providers.base import (
    GenerationRequest,
    MalformedResponse,
    Message,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from llmdrift.providers.litellm_provider import LiteLLMProvider, map_exception
from llmdrift.providers.mock import MockProvider
from llmdrift.providers.registry import build_provider, check_available

REQ = GenerationRequest(
    messages=(Message("system", "be brief"), Message("user", "What is 2+2?")),
    temperature=0.2,
    max_tokens=50,
)


def _cfg(**kw: Any) -> ModelConfig:
    base: dict[str, Any] = {
        "name": "g",
        "provider": "groq",
        "model": "groq/openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY",
        "stream": False,
    }
    base.update(kw)
    return ModelConfig.model_validate(base)


class FakeClock:
    def __init__(self, step: float = 0.1) -> None:
        self.t, self.step = 0.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _response(content: Any, usage: Any = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage
    )


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


@pytest.mark.usefixtures("key")
def test_non_streaming_success_passes_params() -> None:
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response("4", {"prompt_tokens": 12, "completion_tokens": 1})

    result = LiteLLMProvider(_cfg(), fake, clock=FakeClock()).generate(REQ)
    assert result.text == "4"
    assert (result.input_tokens, result.output_tokens) == (12, 1)
    assert not result.tokens_estimated
    assert result.latency_ms == pytest.approx(100.0)
    assert seen["api_key"] == "test-key"
    assert seen["temperature"] == 0.2
    assert seen["num_retries"] == 0
    assert "api_key" not in result.raw


@pytest.mark.usefixtures("key")
def test_streaming_measures_ttft_and_reads_usage() -> None:
    def chunks() -> Iterator[Any]:
        yield {"choices": [{"delta": {"content": ""}}]}
        yield {"choices": [{"delta": {"content": "Hel"}}]}
        yield {"choices": [{"delta": {"content": "lo"}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return chunks()

    result = LiteLLMProvider(_cfg(stream=True), fake, clock=FakeClock()).generate(REQ)
    assert result.text == "Hello"
    assert result.ttft_ms is not None and result.ttft_ms < result.latency_ms
    assert result.output_tokens == 2
    assert seen["stream_options"] == {"include_usage": True}


@pytest.mark.usefixtures("key")
def test_missing_usage_is_estimated() -> None:
    result = LiteLLMProvider(_cfg(), lambda **_: _response("four words in here")).generate(REQ)
    assert result.tokens_estimated
    assert result.input_tokens and result.output_tokens


@pytest.mark.usefixtures("key")
@pytest.mark.parametrize(
    "response",
    [
        _response(""),
        _response(None),
        SimpleNamespace(choices=[], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(message=None)], usage=None),
    ],
)
def test_malformed_responses(response: Any) -> None:
    with pytest.raises(MalformedResponse):
        LiteLLMProvider(_cfg(), lambda **_: response).generate(REQ)


@pytest.mark.usefixtures("key")
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (litellm.RateLimitError("429", "groq", "m"), RateLimited),
        (litellm.Timeout("slow", "m", "groq"), ProviderTimeout),
        (litellm.ServiceUnavailableError("503", "groq", "m"), ProviderUnavailable),
        (litellm.APIConnectionError("down", "groq", "m"), ProviderUnavailable),
        (litellm.InternalServerError("500", "groq", "m"), ProviderUnavailable),
        (litellm.AuthenticationError("401", "groq", "m"), ProviderAuthError),
        (litellm.NotFoundError("404", "m", "groq"), ProviderBadRequest),
        (litellm.BadRequestError("400", "m", "groq"), ProviderBadRequest),
        (TimeoutError("t"), ProviderTimeout),
        (ConnectionError("c"), ProviderUnavailable),
        (KeyError("weird"), ProviderUnavailable),
    ],
)
def test_exception_mapping(exc: Exception, expected: type) -> None:
    def fake(**_: Any) -> Any:
        raise exc

    with pytest.raises(expected):
        LiteLLMProvider(_cfg(), fake).generate(REQ)


def test_map_exception_passthrough() -> None:
    err = RateLimited("x")
    assert map_exception(err) is err


def test_missing_key_is_auth_error() -> None:
    with pytest.raises(ProviderAuthError, match="GROQ_API_KEY"):
        LiteLLMProvider(_cfg(), lambda **_: _response("x")).generate(REQ)


def test_ollama_api_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama:11434")
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response("x", {"prompt_tokens": 1, "completion_tokens": 1})

    cfg = _cfg(provider="ollama", model="ollama_chat/llama3.2:1b", api_key_env=None)
    LiteLLMProvider(cfg, fake).generate(REQ)
    assert seen["api_base"] == "http://ollama:11434"
    assert "api_key" not in seen


def test_check_available(monkeypatch: pytest.MonkeyPatch) -> None:
    assert check_available(_cfg(enabled=False)) == (False, "disabled in config")
    assert check_available(_cfg())[0] is False
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert check_available(_cfg()) == (True, "ok")
    ollama = _cfg(
        provider="ollama", model="ollama_chat/x", api_key_env=None, api_base="http://127.0.0.1:9"
    )
    ok, reason = check_available(ollama, timeout_s=0.5)
    assert not ok and "Ollama not reachable" in reason


def _mock_cfg(**params: Any) -> ModelConfig:
    return ModelConfig(name="m", provider="mock", model="mock/m", params=params)


def test_mock_is_deterministic_and_degrades(dataset: list[GoldenItem]) -> None:
    good = build_provider(_mock_cfg(quality=1.0), dataset)
    bad = build_provider(_mock_cfg(quality=1.0, degrade=1.0), dataset)
    item = next(i for i in dataset if i.accept)
    req = build_request(item, "sys", 0.2, 100, 0)
    first, second = good.generate(req), good.generate(req)
    assert first.text == second.text == item.reference_answer
    assert bad.generate(req).text != item.reference_answer


def test_mock_errors_and_unknown_prompt(dataset: list[GoldenItem]) -> None:
    flaky = MockProvider(_mock_cfg(error_rate=1.0), dataset, render_prompt)
    req = build_request(dataset[0], "sys", 0.2, 100, 0)
    with pytest.raises(ProviderUnavailable):
        flaky.generate(req)
    with pytest.raises(MalformedResponse):
        flaky.generate(REQ)


def test_mock_good_answers_by_behaviour(dataset: list[GoldenItem]) -> None:
    provider = MockProvider(_mock_cfg(quality=1.0), dataset, render_prompt)
    for item in dataset:
        text = provider.generate(build_request(item, "s", 0, 10, 0)).text
        assert text


def test_degradation_prompt_and_grounded_template(dataset: list[GoldenItem]) -> None:
    rag = next(i for i in dataset if i.context)
    req = build_request(rag, "sys", 0.2, 100, 1, inject_degradation=True)
    assert req.messages[0].content == DEGRADED_SYSTEM_PROMPT
    assert rag.context is not None and rag.context in req.messages[1].content
    assert req.repeat_idx == 1
