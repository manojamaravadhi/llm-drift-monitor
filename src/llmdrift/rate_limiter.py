"""Rate limiter and SQLite usage logger for the interactive 'Try it live' tab."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter per visitor session."""

    def __init__(self, max_requests: int = 5, window_seconds: float = 600.0) -> None:
        """Initialize the limiter with request count and window duration in seconds."""
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def check(self, timestamps: list[float], now: float | None = None) -> tuple[bool, int, float]:
        """Check if a new request is allowed.

        Returns ``(allowed, remaining_quota, seconds_until_reset)``.
        """
        current_time = now if now is not None else time.time()
        cutoff = current_time - self.window_seconds
        valid = [t for t in timestamps if t > cutoff]
        remaining = max(0, self.max_requests - len(valid))

        if len(valid) >= self.max_requests:
            oldest = min(valid)
            wait_s = max(0.0, (oldest + self.window_seconds) - current_time)
            return False, 0, wait_s

        return True, remaining, 0.0

    def record(self, timestamps: list[float], now: float | None = None) -> list[float]:
        """Add the current timestamp and prune expired ones."""
        current_time = now if now is not None else time.time()
        cutoff = current_time - self.window_seconds
        valid = [t for t in timestamps if t > cutoff]
        valid.append(current_time)
        return valid


class LiveUsageLogger:
    """Thread-safe SQLite logger for live requests, isolated from evaluation data."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize SQLite database for live telemetry."""
        if db_path is None:
            primary = Path("data/live_usage.db")
            try:
                primary.parent.mkdir(parents=True, exist_ok=True)
                primary.touch(exist_ok=True)
                self.db_path = primary
            except OSError:
                self.db_path = Path("/tmp/live_usage.db")
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prompt_preview TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    ttft_ms REAL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    billed_usd REAL,
                    list_usd REAL,
                    status TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log_request(
        self,
        prompt: str,
        answer: dict[str, Any],
        now: datetime | None = None,
    ) -> None:
        """Record one model's answer telemetry."""
        ts = (now or datetime.now(UTC)).isoformat()
        status = "error" if answer.get("error") else ("skipped" if answer.get("skipped") else "ok")
        preview = prompt[:120] + ("..." if len(prompt) > 120 else "")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO live_requests (
                    timestamp, prompt_preview, model_name, latency_ms, ttft_ms,
                    input_tokens, output_tokens, billed_usd, list_usd, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    preview,
                    str(answer.get("model_name", "unknown")),
                    float(answer.get("latency_ms") or 0.0),
                    float(answer.get("ttft_ms") or 0.0) if answer.get("ttft_ms") else None,
                    int(answer.get("input_tokens") or 0),
                    int(answer.get("output_tokens") or 0),
                    (
                        float(answer.get("billed_usd") or 0.0)
                        if answer.get("billed_usd") is not None
                        else None
                    ),
                    (
                        float(answer.get("list_usd") or 0.0)
                        if answer.get("list_usd") is not None
                        else None
                    ),
                    status,
                ),
            )
            conn.commit()

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate usage stats across all live queries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT COUNT(*) as count FROM live_requests").fetchone()
            total_reqs = row["count"] if row else 0
            if total_reqs == 0:
                return {
                    "total_requests": 0,
                    "model_stats": [],
                    "recent_logs": [],
                }

            model_stats = conn.execute(
                """
                SELECT
                    model_name,
                    COUNT(*) as queries,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(list_usd) as avg_cost_usd,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as error_rate
                FROM live_requests
                GROUP BY model_name
                ORDER BY queries DESC
                """
            ).fetchall()

            recent = conn.execute(
                """
                SELECT timestamp, prompt_preview, model_name, latency_ms, list_usd, status
                FROM live_requests
                ORDER BY id DESC
                LIMIT 10
                """
            ).fetchall()

            return {
                "total_requests": total_reqs,
                "model_stats": [dict(r) for r in model_stats],
                "recent_logs": [dict(r) for r in recent],
            }
