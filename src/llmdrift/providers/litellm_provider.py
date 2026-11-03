"""Real providers (Gemini, Groq, Ollama) behind LiteLLM."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from typing import Any

import litellm
from litellm import exceptions as lx

from llmdrift.config import ModelConfig
from llmdrift.providers.base import (
    CompletionResult,
    GenerationRequest,
    MalformedResponse,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)

litellm.suppress_debug_info = True

CompletionFn = Callable[..., Any]

# Order matters: specific classes before their base classes (e.g. Timeout, APIError).
_EXCEPTION_MAP: tuple[tuple[type[BaseException], type[ProviderError]], ...] = (
    (lx.RateLimitError, RateLimited),
    (lx.Timeout, ProviderTimeout),
    (lx.AuthenticationError, ProviderAuthError),
    (lx.PermissionDeniedError, ProviderAuthError),
    (lx.NotFoundError, ProviderBadRequest),
    (lx.BadRequestError, ProviderBadRequest),
    (lx.ServiceUnavailableError, ProviderUnavailable),
    (lx.InternalServerError, ProviderUnavailable),
    (lx.APIConnectionError, ProviderUnavailable),
    (lx.APIError, ProviderUnavailable),
)


def map_exception(exc: BaseException) -> ProviderError:
    """Translate a LiteLLM / network exception into our error taxonomy."""
    if isinstance(exc, ProviderError):
        return exc
    for source, target in _EXCEPTION_MAP:
        if isinstance(exc, source):
            return target(_short(exc))
    if isinstance(exc, TimeoutError):
        return ProviderTimeout(_short(exc))
    if isinstance(exc, ConnectionError):
        return ProviderUnavailable(_short(exc))
    return ProviderUnavailable(f"{type(exc).__name__}: {_short(exc)}")


def _short(exc: BaseException, limit: int = 300) -> str:
    """First part of an error message (provider errors can be very long)."""
    return str(exc).replace("\n", " ")[:limit]


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict or an attribute-style object."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    if usage is None:
        return None, None
    prompt, completion = _get(usage, "prompt_tokens"), _get(usage, "completion_tokens")
    return (
        int(prompt) if isinstance(prompt, int | float) else None,
        int(completion) if isinstance(completion, int | float) else None,
    )


class LiteLLMProvider:
    """Calls one model through ``litellm.completion``."""

    def __init__(
        self,
        cfg: ModelConfig,
        completion_fn: CompletionFn | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Create the provider. ``completion_fn`` and ``clock`` are injectable for tests."""
        self.cfg = cfg
        self.name = cfg.name
        self.model_id = cfg.model
        self._completion = completion_fn or litellm.completion
        self._clock = clock

    def _kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": request.as_litellm_messages(),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "timeout": self.cfg.timeout_s,
            "num_retries": 0,  # retries are handled by RetryPolicy so they are observable
        }
        if self.cfg.api_key_env:
            key = os.environ.get(self.cfg.api_key_env)
            if not key:
                raise ProviderAuthError(f"environment variable {self.cfg.api_key_env} is not set")
            kwargs["api_key"] = key
        api_base = self.cfg.api_base
        if self.cfg.provider == "ollama":
            api_base = os.environ.get("OLLAMA_API_BASE", api_base)
        if api_base:
            kwargs["api_base"] = api_base
        if self.cfg.stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def generate(self, request: GenerationRequest) -> CompletionResult:
        """Call the model once (no retries). Raises ``ProviderError`` subclasses."""
        kwargs = self._kwargs(request)
        start = self._clock()
        try:
            response = self._completion(**kwargs)
            if self.cfg.stream:
                text, ttft, usage = self._consume_stream(response, start)
            else:
                text, ttft, usage = self._parse_response(response), None, _get(response, "usage")
        except ProviderError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        latency_ms = (self._clock() - start) * 1000.0

        if not text.strip():
            raise MalformedResponse("empty response text")
        in_tok, out_tok = _usage_tokens(usage)
        estimated = False
        if in_tok is None or out_tok is None:
            in_tok, out_tok = self._estimate_tokens(request, text)
            estimated = True
        return CompletionResult(
            text=text,
            latency_ms=latency_ms,
            ttft_ms=ttft,
            input_tokens=in_tok,
            output_tokens=out_tok,
            tokens_estimated=estimated,
            raw={"model": self.cfg.model, "streamed": self.cfg.stream},
        )

    def _parse_response(self, response: Any) -> str:
        choices = _get(response, "choices")
        if not choices:
            raise MalformedResponse("response has no choices")
        message = _get(choices[0], "message")
        content = _get(message, "content") if message is not None else None
        if not isinstance(content, str):
            raise MalformedResponse("response message has no text content")
        return content

    def _consume_stream(self, chunks: Iterable[Any], start: float) -> tuple[str, float | None, Any]:
        parts: list[str] = []
        ttft: float | None = None
        usage: Any = None
        for chunk in chunks:
            chunk_usage = _get(chunk, "usage")
            if chunk_usage is not None:
                usage = chunk_usage
            choices = _get(chunk, "choices")
            if not choices:
                continue
            delta = _get(choices[0], "delta")
            content = _get(delta, "content") if delta is not None else None
            if content:
                if ttft is None:
                    ttft = (self._clock() - start) * 1000.0
                parts.append(str(content))
        return "".join(parts), ttft, usage

    def _estimate_tokens(self, request: GenerationRequest, text: str) -> tuple[int, int]:
        """Tokenizer estimate when the provider does not report usage."""
        try:
            prompt_tokens = litellm.token_counter(
                model=self.cfg.model, messages=request.as_litellm_messages()
            )
            output_tokens = litellm.token_counter(model=self.cfg.model, text=text)
            return int(prompt_tokens), int(output_tokens)
        except Exception:
            words_in = sum(len(m.content.split()) for m in request.messages)
            return round(words_in * 1.3), round(len(text.split()) * 1.3)
