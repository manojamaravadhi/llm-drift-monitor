from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from llmdrift.rate_limiter import LiveUsageLogger, SlidingWindowRateLimiter


def test_sliding_window_rate_limiter() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=100.0)
    timestamps: list[float] = []

    # 1st request
    allowed, remaining, wait_s = limiter.check(timestamps, now=1000.0)
    assert allowed is True
    assert remaining == 3
    assert wait_s == 0.0
    timestamps = limiter.record(timestamps, now=1000.0)

    # 2nd request
    allowed, remaining, _ = limiter.check(timestamps, now=1010.0)
    assert allowed is True
    assert remaining == 2
    timestamps = limiter.record(timestamps, now=1010.0)

    # 3rd request
    allowed, remaining, _ = limiter.check(timestamps, now=1020.0)
    assert allowed is True
    assert remaining == 1
    timestamps = limiter.record(timestamps, now=1020.0)

    # 4th request (should be blocked)
    allowed, remaining, wait_s = limiter.check(timestamps, now=1030.0)
    assert allowed is False
    assert remaining == 0
    assert wait_s == 70.0  # (1000 + 100) - 1030

    # Fast-forward past the 1st request expiration
    allowed, remaining, _ = limiter.check(timestamps, now=1105.0)
    assert allowed is True
    assert remaining == 1


def test_live_usage_logger(tmp_path: Path) -> None:
    db_file = tmp_path / "test_live_usage.db"
    logger = LiveUsageLogger(db_path=db_file)

    # Empty summary
    summary = logger.get_summary()
    assert summary["total_requests"] == 0
    assert summary["model_stats"] == []

    # Record successful request
    logger.log_request(
        prompt="What is AI?",
        answer={
            "model_name": "test-model",
            "latency_ms": 320.5,
            "ttft_ms": 310.0,
            "input_tokens": 15,
            "output_tokens": 25,
            "billed_usd": 0.0,
            "list_usd": 0.000045,
            "error": None,
            "skipped": None,
        },
        now=datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC),
    )

    # Record error request
    logger.log_request(
        prompt="Explain quantum",
        answer={
            "model_name": "test-model",
            "latency_ms": 100.0,
            "error": "rate_limit: slow down",
        },
        now=datetime(2026, 11, 4, 12, 1, 0, tzinfo=UTC),
    )

    summary2 = logger.get_summary()
    assert summary2["total_requests"] == 2
    assert len(summary2["model_stats"]) == 1
    stat = summary2["model_stats"][0]
    assert stat["model_name"] == "test-model"
    assert stat["queries"] == 2
    assert stat["error_rate"] == 0.5
    assert len(summary2["recent_logs"]) == 2
