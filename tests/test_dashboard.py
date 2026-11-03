from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from llmdrift.config import EvalConfig, ModelConfig
from llmdrift.datasets.schema import GoldenItem
from llmdrift.providers.retry import RetryPolicy, Throttle
from llmdrift.runner import Runner, RunOptions
from llmdrift.storage.db import get_engine
from tests.conftest import ROOT

APP = str(ROOT / "dashboard" / "app.py")


def _run(monkeypatch: pytest.MonkeyPatch, db: Path) -> AppTest:
    monkeypatch.setenv("LLMDRIFT_DB", str(db))
    at = AppTest.from_file(APP, default_timeout=120)
    return at.run()


def test_dashboard_renders_runs_with_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    eval_cfg: EvalConfig,
    mock_models: list[ModelConfig],
    degraded_models: list[ModelConfig],
    pricing: dict[str, Any],
    dataset: list[GoldenItem],
) -> None:
    db = tmp_path / "runs.db"
    engine = get_engine(db)
    for models in (mock_models, degraded_models):
        Runner(
            eval_cfg,
            models,
            pricing,
            dataset,
            engine,
            retry_policy=RetryPolicy(sleep=lambda _: None),
            throttle_factory=lambda rpm: Throttle(rpm=1e9, sleep=lambda _: None),
        ).run(RunOptions(fresh=True, use_cache=False))
    engine.dispose()
    at = _run(monkeypatch, db)
    assert not at.exception, at.exception
    assert any("Drift detected" in e.value for e in at.error)
    assert any("Judge" in m.value for m in at.markdown)
    assert len(at.tabs) == 7


def test_dashboard_handles_empty_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    get_engine(db).dispose()
    at = _run(monkeypatch, db)
    assert not at.exception
    assert any("no completed runs" in i.value for i in at.info)


def test_dashboard_handles_missing_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path / "nope.db")
    assert not at.exception
    assert any("No database" in i.value for i in at.info)
