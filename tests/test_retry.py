from __future__ import annotations

import random

import pytest

from llmdrift.providers.base import ProviderBadRequest, RateLimited
from llmdrift.providers.retry import CircuitBreaker, RetryPolicy, Throttle, backoff_delay


def test_backoff_is_bounded_and_grows() -> None:
    rng = random.Random(0)
    for attempt in range(1, 10):
        delay = backoff_delay(attempt, base=1.0, cap=8.0, rng=rng)
        assert 0 <= delay <= min(8.0, 2 ** (attempt - 1))


def test_retry_retries_retryable_errors_then_succeeds() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimited("429")
        return "ok"

    policy = RetryPolicy(max_attempts=5, base_delay_s=1, sleep=sleeps.append, rng=random.Random(1))
    assert policy.call(flaky) == ("ok", 3)
    assert len(sleeps) == 2


def test_retry_gives_up_after_max_attempts() -> None:
    sleeps: list[float] = []
    policy = RetryPolicy(max_attempts=3, sleep=sleeps.append)

    def always() -> str:
        raise RateLimited("429")

    with pytest.raises(RateLimited):
        policy.call(always)
    assert len(sleeps) == 2


def test_non_retryable_error_is_not_retried() -> None:
    sleeps: list[float] = []
    policy = RetryPolicy(max_attempts=5, sleep=sleeps.append)

    def bad() -> str:
        raise ProviderBadRequest("unknown model")

    with pytest.raises(ProviderBadRequest):
        policy.call(bad)
    assert sleeps == []


def test_throttle_spaces_calls() -> None:
    now = {"t": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    throttle = Throttle(rpm=60, clock=lambda: now["t"], sleep=sleep)
    for _ in range(3):
        throttle.wait()
    assert slept == [pytest.approx(1.0), pytest.approx(1.0)]


def test_circuit_breaker() -> None:
    breaker = CircuitBreaker(threshold=2)
    breaker.record(False)
    assert not breaker.is_open
    breaker.record(True)
    breaker.record(False)
    breaker.record(False)
    assert breaker.is_open
