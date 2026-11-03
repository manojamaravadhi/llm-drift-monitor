"""Exponential backoff with full jitter, a client-side throttle and a circuit breaker."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from llmdrift.providers.base import ProviderError

T = TypeVar("T")


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    """Full-jitter delay for a 1-based retry ``attempt``: uniform(0, min(cap, base * 2**(n-1))).

    "Full jitter" is the AWS Architecture Blog variant; it spreads retries from many clients.
    """
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    return rng.uniform(0, ceiling)


@dataclass
class RetryPolicy:
    """Retry retryable ``ProviderError`` subclasses with backoff."""

    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def call(self, fn: Callable[[], T]) -> tuple[T, int]:
        """Run ``fn`` and return ``(result, attempts_used)``. Re-raises the last error."""
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(), attempt
            except ProviderError as exc:
                if not exc.retryable or attempt == self.max_attempts:
                    raise
                self.sleep(backoff_delay(attempt, self.base_delay_s, self.max_delay_s, self.rng))
        raise AssertionError("unreachable")  # pragma: no cover


@dataclass
class Throttle:
    """Keeps calls at or below ``rpm`` requests per minute (minimum spacing)."""

    rpm: float
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _next_allowed: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def wait(self) -> None:
        """Block until the next call is allowed."""
        interval = 60.0 / self.rpm
        with self._lock:
            now = self.clock()
            if now < self._next_allowed:
                self.sleep(self._next_allowed - now)
                now = self._next_allowed
            self._next_allowed = now + interval


@dataclass
class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures so a dead provider is skipped quickly."""

    threshold: int = 5
    consecutive_failures: int = 0

    @property
    def is_open(self) -> bool:
        """True once the provider should no longer be called this run."""
        return self.consecutive_failures >= self.threshold

    def record(self, success: bool) -> None:
        """Update state after a call."""
        self.consecutive_failures = 0 if success else self.consecutive_failures + 1
