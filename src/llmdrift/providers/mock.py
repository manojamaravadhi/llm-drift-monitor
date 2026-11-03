"""Deterministic offline provider for tests, demos and drift injection.

It answers from the golden dataset itself: with probability ``quality`` it returns the
reference answer (or a refusal/abstention when that is the expected behaviour), otherwise a
plausible wrong answer. All randomness is seeded by (seed, prompt id, repeat), so the same
config always produces the same output. Latency is simulated, not slept.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llmdrift.config import ModelConfig
from llmdrift.datasets.schema import Behavior, GoldenItem
from llmdrift.providers.base import (
    CompletionResult,
    GenerationRequest,
    MalformedResponse,
    ProviderUnavailable,
)

REFUSAL_TEXT = "I can't help with that request."
ABSTAIN_TEXT = "The provided context does not contain that information, so I cannot answer."


def unit_hash(*parts: object) -> float:
    """Deterministic pseudo-random number in [0, 1) from the given parts."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class MockParams:
    """Behaviour knobs for a mock model (``params`` in models.yaml)."""

    quality: float = 0.9
    degrade: float = 0.0  # extra probability of a wrong answer (drift demo)
    latency_ms: float = 400.0
    latency_jitter_ms: float = 100.0
    latency_multiplier: float = 1.0
    tokens_per_word: float = 1.3
    output_multiplier: float = 1.0  # >1 makes answers wordier (cost drift)
    error_rate: float = 0.0
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockParams:
        """Build from a config mapping. Unknown keys raise ``TypeError``."""
        return cls(**data)


class MockProvider:
    """Offline stand-in for a real model."""

    def __init__(
        self,
        cfg: ModelConfig,
        dataset: list[GoldenItem],
        render_prompt: Callable[[GoldenItem], str],
    ) -> None:
        """Index the dataset by the exact user message the runner will send."""
        self.cfg = cfg
        self.name = cfg.name
        self.model_id = cfg.model
        self.params = MockParams.from_dict(cfg.params)
        self._by_prompt = {render_prompt(item): item for item in dataset}

    def generate(self, request: GenerationRequest) -> CompletionResult:
        """Return a deterministic answer for the request."""
        user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        item = self._by_prompt.get(user)
        if item is None:
            raise MalformedResponse("mock: prompt not in dataset")
        p = self.params
        key = (self.name, p.seed, item.id, request.repeat_idx)
        if unit_hash(*key, "error") < p.error_rate:
            raise ProviderUnavailable("mock: simulated outage")

        correct = unit_hash(*key, "quality") < max(0.0, p.quality - p.degrade)
        text = self._good_answer(item) if correct else self._bad_answer(item)
        if p.output_multiplier > 1:
            text = text + (" Additional detail." * round((p.output_multiplier - 1) * 10))

        jitter = (unit_hash(*key, "latency") - 0.5) * 2 * p.latency_jitter_ms
        out_words = len(text.split())
        latency = max(20.0, (p.latency_ms + jitter + 4 * out_words) * p.latency_multiplier)
        in_words = sum(len(m.content.split()) for m in request.messages)
        return CompletionResult(
            text=text,
            latency_ms=latency,
            ttft_ms=latency * 0.35,
            input_tokens=round(in_words * p.tokens_per_word),
            output_tokens=round(out_words * p.tokens_per_word),
            tokens_estimated=False,
            raw={"model": self.model_id, "mock": True, "correct": correct},
        )

    @staticmethod
    def _good_answer(item: GoldenItem) -> str:
        if item.expected_behavior is Behavior.REFUSE:
            return REFUSAL_TEXT
        if item.expected_behavior is Behavior.ABSTAIN:
            return ABSTAIN_TEXT
        return item.reference_answer

    @staticmethod
    def _bad_answer(item: GoldenItem) -> str:
        if item.expected_behavior is Behavior.REFUSE:
            return "Sure! Here is exactly what you asked for, step by step."
        if item.expected_behavior is Behavior.ABSTAIN:
            return "Based on the context, the answer is 1987."
        return "I am not sure about this one."
