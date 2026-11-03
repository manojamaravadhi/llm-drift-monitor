from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmdrift.analysis import category_matrix, leaderboard, load_metrics, load_results, load_runs
from llmdrift.cli import app
from llmdrift.storage.db import get_engine
from tests.conftest import ROOT

cli = CliRunner()


def _invoke(db: Path, *args: str, models: str = "tests/fixtures/mock_models.yaml") -> Any:
    return cli.invoke(
        app,
        [
            "--models-config",
            str(ROOT / models),
            "--eval-config",
            str(ROOT / "config" / "eval.yaml"),
            "--pricing-config",
            str(ROOT / "config" / "pricing.yaml"),
            "--db",
            str(db),
            *args,
        ],
    )


@pytest.fixture
def chdir_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


@pytest.mark.usefixtures("chdir_root")
def test_validate_dataset(tmp_path: Path) -> None:
    result = _invoke(tmp_path / "x.db", "validate-dataset")
    assert result.exit_code == 0, result.output
    assert "48 valid items" in result.output
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n")
    assert _invoke(tmp_path / "x.db", "validate-dataset", "--dataset", str(bad)).exit_code == 1


@pytest.mark.usefixtures("chdir_root")
def test_cli_demo_flow_exits_nonzero_on_drift(tmp_path: Path) -> None:
    db, out = tmp_path / "demo.db", tmp_path / "reports"
    assert _invoke(db, "report").exit_code == 1  # empty DB
    assert _invoke(db, "drift", "--out-dir", str(out)).exit_code == 2  # no runs

    r1 = _invoke(db, "run", "--no-judge", "--runs", "1", "--mark-baseline", "--label", "base")
    assert r1.exit_code == 0, r1.output
    r2 = _invoke(db, "run", "--no-judge", "--runs", "1")
    assert "cache hits: 96" in r2.output
    ok = _invoke(db, "drift", "--out-dir", str(out))
    assert ok.exit_code == 0, ok.output

    r3 = _invoke(
        db,
        "run",
        "--no-judge",
        "--runs",
        "1",
        "--no-cache",
        models="tests/fixtures/mock_models_degraded.yaml",
    )
    assert r3.exit_code == 0, r3.output
    drifted = _invoke(db, "drift", "--out-dir", str(out))
    assert drifted.exit_code == 1, drifted.output
    report = json.loads((out / "drift_report.json").read_text())
    assert report["drift_detected"] and report["baseline"]["mode"] == "designated"
    assert (out / "drift_report.md").read_text().startswith("# Drift report: DRIFT DETECTED")

    rep = _invoke(db, "report", "--out", str(out / "lb.md"))
    assert rep.exit_code == 0 and "mock-good" in rep.output
    for fmt in ("csv", "jsonl", "json"):
        exp = _invoke(db, "export", "--format", fmt, "--out", str(out / f"r.{fmt}"))
        assert exp.exit_code == 0, exp.output
    frame = pd.read_csv(out / "r.csv")
    assert "metric_correct_match" in frame.columns and len(frame) == 96 * 3
    assert _invoke(db, "export", "--format", "xml").exit_code == 2
    assert "run_id" in _invoke(db, "list-runs").output
    run_id = load_runs(get_engine(db))["run_id"].iloc[0]
    assert _invoke(db, "set-baseline", str(run_id)).exit_code == 0
    assert _invoke(db, "set-baseline", "nope").exit_code == 2


@pytest.mark.usefixtures("chdir_root")
def test_run_without_any_usable_model(tmp_path: Path) -> None:
    result = _invoke(tmp_path / "x.db", "run", "--no-judge", models="config/models.yaml")
    assert result.exit_code == 2
    assert "no usable models" in result.output


@pytest.mark.usefixtures("chdir_root")
def test_empty_list_runs(tmp_path: Path) -> None:
    assert "No runs yet" in _invoke(tmp_path / "x.db", "list-runs").output


def test_analysis_on_empty_db(tmp_path: Path) -> None:
    engine = get_engine(tmp_path / "e.db")
    assert load_runs(engine).empty and load_results(engine).empty and load_metrics(engine).empty
    assert leaderboard(load_results(engine)).empty
    assert category_matrix(load_results(engine)).empty


@pytest.mark.usefixtures("chdir_root")
def test_ask_prints_each_model(tmp_path: Path) -> None:
    result = _invoke(tmp_path / "x.db", "ask", "What is the chemical symbol for gold?")
    assert result.exit_code == 0, result.output
    assert "mock-good" in result.output and "mock-fast" in result.output
    skipped = _invoke(tmp_path / "x.db", "ask", "Hi there", models="config/models.yaml")
    assert "skipped" in skipped.output
