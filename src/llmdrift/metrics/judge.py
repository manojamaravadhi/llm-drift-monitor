"""LLM-as-judge metrics via DeepEval, with any LiteLLM model as the judge.

Judge scores are noisy (they vary between calls even at temperature 0) and a judge from the
same family as a candidate tends to favour it (self-preference bias). They are therefore
combined with deterministic checks, never used alone.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from llmdrift.config import JudgeConfig
from llmdrift.datasets.schema import Behavior, GoldenItem
from llmdrift.metrics.deterministic import MetricValue
from llmdrift.providers.retry import Throttle

CORRECTNESS_CRITERIA = (
    "Determine whether the actual output is factually correct and consistent with the "
    "expected output. Paraphrasing is fine; contradictions, missing key facts or "
    "invented facts are not."
)


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
    """Runs DeepEval GEval / AnswerRelevancy / Faithfulness with a LiteLLM judge model."""

    def __init__(self, cfg: JudgeConfig, throttle: Throttle | None = None) -> None:
        """Build DeepEval metric objects once. Imports DeepEval lazily (it is slow)."""
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
        from deepeval.models import LiteLLMModel
        from deepeval.test_case import SingleTurnParams

        self.cfg = cfg
        self._throttle = throttle or Throttle(rpm=cfg.rpm)
        api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        model = LiteLLMModel(model=cfg.model, api_key=api_key, temperature=cfg.temperature)
        self._metrics: dict[str, Any] = {}
        if "correctness" in cfg.metrics:
            self._metrics["correctness"] = GEval(
                name="Correctness",
                criteria=CORRECTNESS_CRITERIA,
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                model=model,
                async_mode=False,
            )
        if "relevancy" in cfg.metrics:
            self._metrics["relevancy"] = AnswerRelevancyMetric(model=model, async_mode=False)
        if "faithfulness" in cfg.metrics:
            self._metrics["faithfulness"] = FaithfulnessMetric(model=model, async_mode=False)

    def _applicable(self, item: GoldenItem) -> list[str]:
        if item.expected_behavior is not Behavior.ANSWER:
            return []  # refusals/abstentions are scored deterministically
        names: list[str] = [n for n in ("correctness", "relevancy") if n in self._metrics]
        if item.context and "faithfulness" in self._metrics:
            names.append("faithfulness")
        return names

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
        for name in self._applicable(item):
            self._throttle.wait()
            metric = self._metrics[name]
            try:
                metric.measure(case)
                value = float(metric.score) if metric.score is not None else None
                out.append(MetricValue(f"judge_{name}", value, "judge", str(metric.reason or "")))
            except Exception as exc:  # judge errors must never fail the run
                reason = f"judge error: {type(exc).__name__}: {str(exc)[:200]}"
                out.append(MetricValue(f"judge_{name}", None, "judge", reason))
        return out
