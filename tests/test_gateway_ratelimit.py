"""Unit tests for the per-key sliding-window limiter.

The clock is injected throughout, so the window boundary is crossed by
arithmetic rather than by sleeping for a minute.
"""

import pytest

from conduit.gateway.ratelimit import RateLimiter


class Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_requests_under_the_limit_are_admitted() -> None:
    limiter = RateLimiter(clock=Clock())
    for _ in range(3):
        assert limiter.check("key", 3).allowed


def test_the_request_over_the_limit_is_refused() -> None:
    limiter = RateLimiter(clock=Clock())
    for _ in range(3):
        limiter.check("key", 3)
    decision = limiter.check("key", 3)
    assert not decision.allowed
    assert decision.remaining == 0


def test_a_refusal_carries_an_honest_retry_after() -> None:
    clock = Clock()
    limiter = RateLimiter(window_s=60.0, clock=clock)
    limiter.check("key", 1)
    clock.advance(20.0)

    decision = limiter.check("key", 1)
    assert not decision.allowed
    # The oldest hit was 20s ago, so a slot frees in the remaining 40s.
    assert decision.retry_after_s == 40
    assert decision.headers()["Retry-After"] == "40"


def test_retry_after_is_never_zero() -> None:
    """A zero would invite an immediate retry that is guaranteed to fail."""
    clock = Clock()
    limiter = RateLimiter(window_s=60.0, clock=clock)
    limiter.check("key", 1)
    clock.advance(59.999)
    assert limiter.check("key", 1).retry_after_s >= 1


def test_the_window_slides() -> None:
    clock = Clock()
    limiter = RateLimiter(window_s=60.0, clock=clock)
    limiter.check("key", 1)
    assert not limiter.check("key", 1).allowed

    clock.advance(60.1)
    assert limiter.check("key", 1).allowed


def test_keys_do_not_share_a_budget() -> None:
    limiter = RateLimiter(clock=Clock())
    limiter.check("a", 1)
    assert not limiter.check("a", 1).allowed
    assert limiter.check("b", 1).allowed


def test_the_limit_is_per_call_so_keys_can_differ() -> None:
    """Two keys with different quotas share one limiter instance."""
    limiter = RateLimiter(clock=Clock())
    assert limiter.check("fast", 5).allowed
    assert limiter.check("fast", 5).remaining == 3
    limiter.check("slow", 1)
    assert not limiter.check("slow", 1).allowed


def test_headers_omit_retry_after_when_admitted() -> None:
    headers = RateLimiter(clock=Clock()).check("key", 10).headers()
    assert "Retry-After" not in headers
    assert headers["X-RateLimit-Limit"] == "10"
    assert headers["X-RateLimit-Remaining"] == "9"


def test_a_nonsense_limit_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RateLimiter(clock=Clock()).check("key", 0)


def test_reset_clears_one_key_or_all_of_them() -> None:
    limiter = RateLimiter(clock=Clock())
    limiter.check("a", 1)
    limiter.check("b", 1)
    limiter.reset("a")
    assert limiter.check("a", 1).allowed
    assert not limiter.check("b", 1).allowed
    limiter.reset()
    assert limiter.check("b", 1).allowed
