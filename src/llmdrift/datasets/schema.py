"""Golden dataset schema and loader."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class Category(StrEnum):
    """Balanced test categories."""

    FACTUAL = "factual"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    INSTRUCTION = "instruction"
    RAG = "rag"
    SAFETY = "safety"


class Behavior(StrEnum):
    """What a good response does."""

    ANSWER = "answer"
    REFUSE = "refuse"
    ABSTAIN = "abstain"  # grounded question whose answer is not in the context


class _Check(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegexCheck(_Check):
    """Response must match a regular expression."""

    type: Literal["regex"]
    pattern: str
    flags: str = ""

    @model_validator(mode="after")
    def _compiles(self) -> RegexCheck:
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex {self.pattern!r}: {exc}") from exc
        return self


class JsonCheck(_Check):
    """Response must be valid JSON, optionally with required keys."""

    type: Literal["json"]
    required_keys: list[str] = Field(default_factory=list)


class MaxWordsCheck(_Check):
    """Response must have at most ``value`` words."""

    type: Literal["max_words"]
    value: int = Field(gt=0)


class BulletCountCheck(_Check):
    """Response must have exactly ``value`` bullet or numbered lines."""

    type: Literal["bullet_count"]
    value: int = Field(gt=0)


class ContainsAllCheck(_Check):
    """Every value must appear (case-insensitive). Score = fraction present."""

    type: Literal["contains_all"]
    values: list[str] = Field(min_length=1)


class NotContainsCheck(_Check):
    """None of the values may appear (case-insensitive)."""

    type: Literal["not_contains"]
    values: list[str] = Field(min_length=1)


class CaseCheck(_Check):
    """Letters must all be upper or lower case."""

    type: Literal["all_caps", "lowercase"]


Check = Annotated[
    RegexCheck
    | JsonCheck
    | MaxWordsCheck
    | BulletCountCheck
    | ContainsAllCheck
    | NotContainsCheck
    | CaseCheck,
    Field(discriminator="type"),
]


class GoldenItem(BaseModel):
    """One test case of the golden dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]+-\d{3}$")
    category: Category
    prompt: str = Field(min_length=5)
    reference_answer: str = Field(min_length=1)
    context: str | None = None
    expected_behavior: Behavior = Behavior.ANSWER
    accept: list[str] = Field(
        default_factory=list,
        description="Strings; a response containing any of them is correct (normalised).",
    )
    checks: list[Check] = Field(default_factory=list)
    source: str = "hand-written"

    @model_validator(mode="after")
    def _consistent(self) -> GoldenItem:
        if self.category is Category.RAG and not self.context:
            raise ValueError(f"{self.id}: rag items need a context")
        if self.expected_behavior is Behavior.ABSTAIN and not self.context:
            raise ValueError(f"{self.id}: abstain items need a context")
        guarded = any(isinstance(c, NotContainsCheck) for c in self.checks)
        if (
            self.category is Category.SAFETY
            and self.expected_behavior is not Behavior.REFUSE
            and not guarded
        ):
            raise ValueError(f"{self.id}: safety items must expect a refusal or use not_contains")
        has_signal = self.accept or self.checks or self.expected_behavior is not Behavior.ANSWER
        if not has_signal:
            raise ValueError(f"{self.id}: needs accept strings or checks for deterministic scoring")
        return self


_items_adapter = TypeAdapter(list[GoldenItem])


def load_dataset(path: str | Path) -> list[GoldenItem]:
    """Load and validate a JSONL golden dataset. Raises ``ValueError`` on any problem."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON: {exc}") from exc
    items = _items_adapter.validate_python(rows)
    ids = [i.id for i in items]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate ids: {dupes}")
    return items
