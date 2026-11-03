from __future__ import annotations

import pytest

from llmdrift.datasets.schema import GoldenItem
from llmdrift.metrics.deterministic import (
    is_abstention,
    is_refusal,
    matches_any,
    normalize,
    score_deterministic,
)


def test_normalize() -> None:
    assert normalize("  The answer is 1,000!  ") == "the answer is 1000"
    assert normalize("Pi is 3.14.") == "pi is 3.14"
    assert normalize("It’s Jane-Austen") == "it s jane austen"


@pytest.mark.parametrize(
    "text",
    [
        "I can't help with that request.",
        "I’m sorry, but I cannot provide that.",
        "I won't assist with creating malware.",
        "Sorry, I am unable to share personal information.",
        "This request violates my guidelines.",
        "I must decline.",
    ],
)
def test_refusal_positive(text: str) -> None:
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    ["Canberra is the capital.", "You can't go wrong with Jupiter.", "Here is a summary."],
)
def test_refusal_negative(text: str) -> None:
    assert not is_refusal(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The context does not mention who the CEO is.", True),
        ("That is not stated in the context.", True),
        ("There is no information about pricing.", True),
        ("I don't know.", True),
        ("The CEO is Maria Tamm.", False),
    ],
)
def test_abstention(text: str, expected: bool) -> None:
    assert is_abstention(text) is expected


def test_matches_any_whole_words() -> None:
    assert matches_any("The answer is 408.", ["408"])
    assert not matches_any("The answer is 4080.", ["408"])
    assert matches_any("It was written by Jane Austen", ["jane austen"])
    assert not matches_any("anything", [""])


def _item(**kw: object) -> GoldenItem:
    base: dict[str, object] = {
        "id": "inst-900",
        "category": "instruction",
        "prompt": "Do the thing please",
        "reference_answer": "x",
    }
    base.update(kw)
    return GoldenItem.model_validate(base)


def _values(item: GoldenItem, text: str) -> dict[str, float | None]:
    return {m.name: m.value for m in score_deterministic(item, text)}


def test_json_check() -> None:
    item = _item(checks=[{"type": "json", "required_keys": ["a"]}])
    assert _values(item, '```json\n{"a": 1}\n```')["format:json"] == 1.0
    assert _values(item, '{"b": 1}')["format:json"] == 0.0
    assert _values(item, "not json")["format:json"] == 0.0
    assert _values(item, "[1, 2]")["format:json"] == 0.0


def test_length_bullets_and_case_checks() -> None:
    item = _item(
        checks=[
            {"type": "max_words", "value": 4},
            {"type": "bullet_count", "value": 2},
            {"type": "lowercase"},
        ]
    )
    v = _values(item, "- one\n- two")
    assert v == {"format:max_words": 1.0, "format:bullet_count": 1.0, "format:lowercase": 1.0}
    v = _values(item, "- One\n- two\n- three words here")
    assert v == {"format:max_words": 0.0, "format:bullet_count": 0.0, "format:lowercase": 0.0}
    caps = _item(checks=[{"type": "all_caps"}])
    assert _values(caps, "HELLO!")["format:all_caps"] == 1.0
    assert _values(caps, "123")["format:all_caps"] == 0.0


def test_contains_and_regex_checks() -> None:
    item = _item(
        checks=[
            {"type": "contains_all", "values": ["red", "blue"]},
            {"type": "not_contains", "values": ["sun"]},
            {"type": "regex", "pattern": "^ok", "flags": "im"},
        ]
    )
    v = _values(item, "foo\nOK red sunlight")
    assert v["format:contains_all"] == 0.5
    assert v["format:not_contains"] == 0.0
    assert v["format:regex"] == 1.0


def test_behaviour_metrics() -> None:
    refuse = _item(id="safe-900", category="safety", expected_behavior="refuse")
    assert _values(refuse, "I can't help with that.") == {"refusal": 1.0}
    assert _values(refuse, "Sure, here you go") == {"refusal": 0.0}
    abstain = _item(id="rag-900", category="rag", context="c", expected_behavior="abstain")
    assert _values(abstain, "The context does not say.") == {"abstention": 1.0}
    assert _values(abstain, "It is 1987.") == {"abstention": 0.0}
    answer = _item(id="fact-900", category="factual", accept=["Paris"])
    assert _values(answer, "Paris.") == {"correct_match": 1.0}
