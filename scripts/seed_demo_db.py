"""Generate data/demo.db: 8 weeks of SYNTHETIC history for the dashboard demo.

No API calls are made. Three offline mock models stand in for the real ones; their quality,
latency and verbosity profiles are invented. Each run goes through the real runner and
scoring code, so the dashboard and drift detector see exactly the shape of real data.
In the last two weeks "demo-groq-gpt-oss-20b" is degraded so an active drift alert appears.

Every run is stored with ``synthetic=True`` and a "SYNTHETIC" label.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from llmdrift.config import ModelConfig, load_eval_config, load_pricing
from llmdrift.datasets.schema import load_dataset
from llmdrift.providers.retry import RetryPolicy, Throttle
from llmdrift.runner import Runner, RunOptions
from llmdrift.storage.db import get_engine, set_baseline

ROOT = Path(__file__).resolve().parents[1]
WEEKS = 8
DEGRADE_FROM_WEEK = 6  # 0-based: weeks 6 and 7 are degraded
START = datetime(2026, 9, 8, 6, 0, tzinfo=UTC)

PROFILES: dict[str, dict[str, Any]] = {
    # name: (model id for pricing, mock params)
    "demo-gemini-flash-lite": {
        "model": "mock/good",
        "params": {"quality": 0.90, "latency_ms": 650, "latency_jitter_ms": 200},
    },
    "demo-groq-gpt-oss-20b": {
        "model": "mock/fast",
        "params": {"quality": 0.86, "latency_ms": 260, "latency_jitter_ms": 80, "error_rate": 0.01},
    },
    "demo-ollama-llama3.2-1b": {
        "model": "mock/local",  # no list price: local model
        "params": {"quality": 0.62, "latency_ms": 1300, "latency_jitter_ms": 400},
    },
}


def models_for_week(week: int) -> list[ModelConfig]:
    """Model configs for one synthetic week (seeded noise, degradation at the end)."""
    out = []
    for name, profile in PROFILES.items():
        params = {**profile["params"], "seed": week}
        if name == "demo-groq-gpt-oss-20b" and week >= DEGRADE_FROM_WEEK:
            params.update(degrade=0.30, latency_multiplier=1.7, output_multiplier=1.4)
        out.append(
            ModelConfig(name=name, provider="mock", model=profile["model"], rpm=1e6, params=params)
        )
    return out


def main() -> None:
    """Create the demo database from scratch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT / "data" / "demo.db"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    db = Path(args.db)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    engine = get_engine(db)
    cfg = load_eval_config(ROOT / "config" / "eval.yaml")
    dataset = load_dataset(ROOT / "datasets" / "golden.jsonl")
    pricing = load_pricing(ROOT / "config" / "pricing.yaml")

    for week in range(WEEKS):
        runner = Runner(
            cfg,
            models_for_week(week),
            pricing,
            dataset,
            engine,
            retry_policy=RetryPolicy(max_attempts=2, sleep=lambda _: None),
            throttle_factory=lambda rpm: Throttle(rpm=rpm, sleep=lambda _: None),
        )
        degraded = week >= DEGRADE_FROM_WEEK
        summary = runner.run(
            RunOptions(
                fresh=True,
                use_cache=False,
                synthetic=True,
                started_at=START + timedelta(weeks=week),
                label=f"SYNTHETIC demo week {week + 1}" + (" (degraded)" if degraded else ""),
                run_id=f"demo-week-{week + 1:02d}",
            )
        )
        print(f"{summary.run_id}: {summary.calls_made} simulated calls")

    with Session(engine) as session:  # last healthy week is the designated baseline
        set_baseline(session, f"demo-week-{DEGRADE_FROM_WEEK:02d}")
        session.commit()

    with engine.connect() as conn:  # fold the WAL into the main file so one file ships
        conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.exec_driver_sql("PRAGMA journal_mode=DELETE")
    engine.dispose()
    print(f"wrote {db}")


if __name__ == "__main__":
    main()
