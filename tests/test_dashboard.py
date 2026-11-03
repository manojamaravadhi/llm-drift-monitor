from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from llmdrift.storage.db import get_engine
from tests.conftest import ROOT

APP = str(ROOT / "dashboard" / "app.py")


def _run(monkeypatch: pytest.MonkeyPatch, db: Path) -> AppTest:
    monkeypatch.setenv("LLMDRIFT_DB", str(db))
    at = AppTest.from_file(APP, default_timeout=120)
    return at.run()


def test_dashboard_renders_demo_db(monkeypatch: pytest.MonkeyPatch) -> None:
    demo = ROOT / "data" / "demo.db"
    if not demo.exists():
        pytest.skip("demo DB not generated (make demo-db)")
    at = _run(monkeypatch, demo)
    assert not at.exception, at.exception
    assert any("Synthetic demo data" in w.value for w in at.warning)
    assert any("Drift detected" in e.value for e in at.error)
    assert len(at.tabs) == 6


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
