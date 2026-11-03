from __future__ import annotations

from typing import Any

import pytest

from llmdrift.config import JudgeConfig
from llmdrift.datasets.schema import GoldenItem
from llmdrift.metrics.judge import DeepEvalJudge, NullJudge, judge_available
from llmdrift.providers.retry import Throttle


def test_judge_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    assert judge_available(JudgeConfig(enabled=False))[0] is False
    assert judge_available(JudgeConfig())[0] is False
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert judge_available(JudgeConfig()) == (True, "ok")


def test_null_judge(dataset: list[GoldenItem]) -> None:
    assert NullJudge().score(dataset[0], "p", "r") == []


class FakeMetric:
    def __init__(self, score: float | None = 0.8, fail: bool = False) -> None:
        self.score, self.reason, self.fail, self.cases = score, "because", fail, []

    def measure(self, case: Any) -> None:
        if self.fail:
            raise RuntimeError("judge quota exhausted")
        self.cases.append(case)


def test_deepeval_judge_applies_metrics_and_survives_errors(
    monkeypatch: pytest.MonkeyPatch, dataset: list[GoldenItem]
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg = JudgeConfig(metrics=["correctness", "relevancy", "faithfulness"])
    sleeps: list[float] = []
    judge = DeepEvalJudge(
        cfg, throttle=Throttle(rpm=1e9, sleep=lambda _: None), sleep=sleeps.append
    )
    assert set(judge._metrics) == {"correctness", "relevancy", "faithfulness"}
    judge._metrics = {
        "correctness": FakeMetric(0.8),
        "relevancy": FakeMetric(fail=True),
        "faithfulness": FakeMetric(1.0),
    }
    rag = next(i for i in dataset if i.context and i.expected_behavior.value == "answer")
    out = {m.name: m for m in judge.score(rag, "prompt", "Tallinn")}
    assert out["judge_correctness"].value == 0.8
    assert out["judge_relevancy"].value is None
    assert "quota" in out["judge_relevancy"].reason
    assert out["judge_faithfulness"].value == 1.0
    assert sleeps == [5.0, 10.0]  # failing metric retried twice before giving up

    factual = next(i for i in dataset if i.category.value == "factual")
    assert "judge_faithfulness" not in {m.name for m in judge.score(factual, "p", "r")}
    refuse = next(i for i in dataset if i.expected_behavior.value == "refuse")
    assert judge.score(refuse, "p", "I can't") == []
