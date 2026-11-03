"""Provider interface and shared result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    """Everything that determines a model response (and therefore the cache key)."""

    messages: tuple[Message, ...]
    temperature: float
    max_tokens: int
    repeat_idx: int = 0

    def as_litellm_messages(self) -> list[dict[str, str]]:
        """Messages in OpenAI/LiteLLM format."""
        return [{"role": m.role, "content": m.content} for m in self.messages]


@dataclass
class CompletionResult:
    """Normalised response from any provider."""

    text: str
    latency_ms: float
    ttft_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_estimated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    """Base class for provider failures that are recorded, not raised to the user."""

    kind = "provider_error"
    retryable = False


class RateLimited(ProviderError):
    """HTTP 429 or quota exhausted."""

    kind = "rate_limit"
    retryable = True


class ProviderTimeout(ProviderError):
    """Request took longer than the configured timeout."""

    kind = "timeout"
    retryable = True


class ProviderUnavailable(ProviderError):
    """Outage, 5xx, or connection error."""

    kind = "unavailable"
    retryable = True


class MalformedResponse(ProviderError):
    """Response had no usable text or an unexpected shape."""

    kind = "malformed_response"
    retryable = True


class ProviderAuthError(ProviderError):
    """Missing or invalid credentials."""

    kind = "auth"


class ProviderBadRequest(ProviderError):
    """Invalid request or unknown model. Retrying will not help."""

    kind = "bad_request"


class Provider(Protocol):
    """Anything that can turn a request into a completion."""

    name: str
    model_id: str

    def generate(self, request: GenerationRequest) -> CompletionResult:
        """Return one completion or raise a ``ProviderError``."""
        ...
