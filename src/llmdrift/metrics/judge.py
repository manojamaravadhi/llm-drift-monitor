"""LLM-as-judge metrics via DeepEval, with any LiteLLM model as the judge.

Judge scores are noisy (they vary between calls even at temperature 0) and a judge from the
same family as a candidate tends to favour it (self-preference bias). They are therefore
combined with deterministic checks, never used alone.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from llmdrift.config import JudgeConfig
from llmdrift.datasets.schema import Behavior, GoldenItem
from llmdrift.metrics.deterministic import MetricValue
from llmdrift.providers.retry import Throttle

CORRECTNESS_STEPS = [
    "Compare the facts in 'actual output' with the 'expected output' for the given 'input'.",
    "Paraphrasing, extra formatting and harmless extra detail are fine.",
    "Heavily penalize contradictions with the expected output and invented facts.",
    "Penalize missing key facts, and ignoring format or length instructions in the input.",
]

FAITHFULNESS_STEPS = [
    "Identify every factual claim in 'actual output'.",
    "Check each claim against 'retrieval context'. Claims not supported by it are "
    "hallucinations, even if true in the real world.",
    "Score 1 when all claims are supported, 0 when the main answer is unsupported.",
]


class Judge(Protocol):
    """Scores one response with LLM-based metrics."""

    def score(self, item: GoldenItem, prompt: str, response: str) -> list[MetricValue]:
        """Return judge metric values for this response."""
        ...


class NullJudge:
    """Used when judging is disabled or unavailable."""

    def score(self, item: GoldenItem, prompt: str, response: str) -> list[MetricValue]:
        """Return no metrics."""
        return []


def judge_available(cfg: JudgeConfig) -> tuple[bool, str]:
    """Whether the judge can run (enabled, key present)."""
    if not cfg.enabled:
        return False, "judge disabled in config"
    if cfg.api_key_env and not os.environ.get(cfg.api_key_env):
        return False, f"{cfg.api_key_env} not set; judge metrics skipped"
    return True, "ok"


class DeepEvalJudge:
    """Runs DeepEval metrics with any LiteLLM model as the judge.

    Built for free-tier quotas: correctness and faithfulness are G-Eval metrics with fixed
    evaluation steps, so each costs one judge call per answer (DeepEval would otherwise spend
    extra calls generating steps and reasons). Answer relevancy is optional and costs more.
    """

    def __init__(
        self,
        cfg: JudgeConfig,
        throttle: Throttle | None = None,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Build DeepEval metric objects once. Imports DeepEval lazily (it is slow)."""
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
        import litellm
        from deepeval.metrics import AnswerRelevancyMetric, GEval
        from deepeval.models import LiteLLMModel
        from deepeval.test_case import SingleTurnParams

        # G-Eval asks for token log-probabilities, which e.g. Gemini does not support.
        litellm.drop_params = True
        self.cfg = cfg
        self._throttle = throttle or Throttle(rpm=cfg.rpm)
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._lock = threading.Lock()  # metric objects hold state; one judgement at a time
        api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        model = LiteLLMModel(
            model=cfg.model, api_key=api_key, base_url=cfg.api_base, temperature=cfg.temperature
        )
        self._metrics: dict[str, Any] = {}
        if "correctness" in cfg.metrics:
            self._metrics["correctness"] = GEval(
                name="Correctness",
                evaluation_steps=CORRECTNESS_STEPS,
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                model=model,
                async_mode=False,
            )
        if "faithfulness" in cfg.metrics:
            self._metrics["faithfulness"] = GEval(
                name="Faithfulness",
                evaluation_steps=FAITHFULNESS_STEPS,
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.RETRIEVAL_CONTEXT,
                ],
                model=model,
                async_mode=False,
            )
        if "relevancy" in cfg.metrics:
            self._metrics["relevancy"] = AnswerRelevancyMetric(
                model=model, async_mode=False, include_reason=False
            )

    def _applicable(self, item: GoldenItem) -> list[str]:
        if item.expected_behavior is not Behavior.ANSWER:
            return []  # refusals/abstentions are scored deterministically
        names: list[str] = [n for n in ("correctness", "relevancy") if n in self._metrics]
        if item.context and "faithfulness" in self._metrics:
            names.append("faithfulness")
        return names

    def _measure(self, metric: Any, case: Any) -> None:
        """Measure with retries: free-tier judges often return 429/503 for a few seconds."""
        for attempt in range(1, self._max_attempts + 1):
            self._throttle.wait()
            try:
                metric.measure(case)
                return
            except Exception:
                if attempt == self._max_attempts:
                    raise
                self._sleep(5.0 * attempt)

    def score(self, item: GoldenItem, prompt: str, response: str) -> list[MetricValue]:
        """Score one response. Judge failures become ``value=None`` with the error as reason."""
        from deepeval.test_case import LLMTestCase

        case = LLMTestCase(
            input=prompt,
            actual_output=response,
            expected_output=item.reference_answer,
            retrieval_context=[item.context] if item.context else None,
        )
        out: list[MetricValue] = []
        with self._lock:
            out.extend(self._score_locked(item, case))
        return out

    def _score_locked(self, item: GoldenItem, case: Any) -> list[MetricValue]:
        out: list[MetricValue] = []
        for name in self._applicable(item):
            metric = self._metrics[name]
            try:
                self._measure(metric, case)
                value = float(metric.score) if metric.score is not None else None
                out.append(MetricValue(f"judge_{name}", value, "judge", str(metric.reason or "")))
            except Exception as exc:  # judge errors must never fail the run
                reason = f"judge error: {type(exc).__name__}: {str(exc)[:200]}"
                out.append(MetricValue(f"judge_{name}", None, "judge", reason))
        return out
