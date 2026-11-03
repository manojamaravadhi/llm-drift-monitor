from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmdrift.datasets.schema import Behavior, Category, GoldenItem, load_dataset
from llmdrift.metrics.deterministic import score_deterministic
from llmdrift.metrics.scoring import quality_score


def test_golden_size_and_balance(dataset: list[GoldenItem]) -> None:
    assert 40 <= len(dataset) <= 60
    counts = Counter(i.category for i in dataset)
    assert set(counts) == set(Category)
    assert max(counts.values()) - min(counts.values()) <= 2


def test_golden_covers_hallucination_and_safety(dataset: list[GoldenItem]) -> None:
    behaviours = Counter(i.expected_behavior for i in dataset)
    assert behaviours[Behavior.ABSTAIN] >= 3
    assert behaviours[Behavior.REFUSE] >= 5
    injections = [i for i in dataset if i.category is Category.SAFETY and "IGNORE" in i.prompt]
    assert injections, "needs at least one prompt-injection item"


def test_reference_answers_pass_their_own_checks(dataset: list[GoldenItem]) -> None:
    """Guards against typos in accept lists / regexes: the reference must score 1.0."""
    for item in dataset:
        metrics = score_deterministic(item, item.reference_answer)
        assert metrics, item.id
        assert quality_score(metrics) == 1.0, (item.id, metrics)


def _item(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fact-999",
        "category": "factual",
        "prompt": "What is two plus two?",
        "reference_answer": "4",
        "accept": ["4"],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "Bad_ID"}, "pattern"),
        ({"category": "rag"}, "context"),
        ({"category": "safety", "expected_behavior": "answer"}, "refusal"),
        ({"expected_behavior": "abstain"}, "context"),
        ({"accept": []}, "deterministic"),
        ({"checks": [{"type": "regex", "pattern": "("}]}, "invalid regex"),
        ({"checks": [{"type": "nope"}]}, None),
        ({"surprise": 1}, None),
    ],
)
def test_invalid_items_rejected(overrides: dict[str, object], message: str | None) -> None:
    with pytest.raises((ValidationError, ValueError), match=message):
        GoldenItem.model_validate(_item(**overrides))


def test_loader_rejects_duplicates_and_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    row = json.dumps(_item())
    p.write_text(row + "\n\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(p)
    p.write_text("{not json}\n")
    with pytest.raises(ValueError, match="line 1"):
        load_dataset(p)
