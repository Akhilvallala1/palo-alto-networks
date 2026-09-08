"""Circuit-breaker state machine (AC-4, AC-5).

The clock is injected, so "half-opens after 30s" is asserted at the exact
boundary rather than approximated by a sleep.
"""

import pytest

from conduit.providers.failover import (
    FAILURE_THRESHOLD,
    RECOVERY_SECONDS,
    BreakerState,
    CircuitBreaker,
    CircuitBreakers,
)


class Clock:
    """A monotonic clock the test drives by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def breaker(clock: Clock) -> CircuitBreaker:
    return CircuitBreaker(clock=clock)


def test_the_documented_defaults_are_five_failures_and_thirty_seconds() -> None:
    assert FAILURE_THRESHOLD == 5
    assert RECOVERY_SECONDS == 30.0
    default = CircuitBreaker()
    assert (default.threshold, default.recovery_s) == (5, 30.0)


def test_a_fresh_breaker_is_closed_and_allows_calls(breaker: CircuitBreaker) -> None:
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allows() is True


def test_four_consecutive_failures_leave_it_closed(breaker: CircuitBreaker) -> None:
    """AC-4: 'exactly 5' means four is not enough."""
    for _ in range(4):
        breaker.record_failure()
    assert breaker.consecutive_failures == 4
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allows() is True


def test_the_fifth_consecutive_failure_opens_it(breaker: CircuitBreaker) -> None:
    """AC-4."""
    for _ in range(5):
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.allows() is False


def test_an_open_breaker_rejects_fast_for_the_whole_window(
    breaker: CircuitBreaker, clock: Clock
) -> None:
    """AC-4: rejects fast while open."""
    for _ in range(5):
        breaker.record_failure()
    for elapsed in (0.0, 1.0, 15.0, 29.999):
        clock.now = 1000.0 + elapsed
        assert breaker.allows() is False, f"allowed a call {elapsed}s into the open window"


def test_a_success_resets_the_consecutive_counter(breaker: CircuitBreaker) -> None:
    """Four failures, a success, four more failures: still closed."""
    for _ in range(4):
        breaker.record_failure()
    breaker.record_success()
    assert breaker.consecutive_failures == 0
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED


def test_non_consecutive_failures_never_open_it(breaker: CircuitBreaker) -> None:
    for _ in range(20):
        breaker.record_failure()
        breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_it_half_opens_exactly_at_thirty_seconds(breaker: CircuitBreaker, clock: Clock) -> None:
    """AC-5."""
    for _ in range(5):
        breaker.record_failure()
    clock.advance(29.999)
    assert breaker.state is BreakerState.OPEN
    clock.advance(0.001)
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allows() is True


def test_a_successful_probe_closes_it(breaker: CircuitBreaker, clock: Clock) -> None:
    """AC-5: half-open probe closes the breaker on success."""
    for _ in range(5):
        breaker.record_failure()
    clock.advance(30.0)
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.consecutive_failures == 0


def test_a_failed_probe_reopens_it_for_a_fresh_window(
    breaker: CircuitBreaker, clock: Clock
) -> None:
    """AC-5: reopens on failure, and the 30s clock restarts."""
    for _ in range(5):
        breaker.record_failure()
    clock.advance(30.0)
    assert breaker.state is BreakerState.HALF_OPEN

    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    clock.advance(29.999)
    assert breaker.state is BreakerState.OPEN
    clock.advance(0.001)
    assert breaker.state is BreakerState.HALF_OPEN


def test_probe_failures_do_not_need_five_more_to_reopen(
    breaker: CircuitBreaker, clock: Clock
) -> None:
    """One failed probe is enough; it does not silently allow four more calls."""
    for _ in range(5):
        breaker.record_failure()
    for _ in range(3):
        clock.advance(30.0)
        assert breaker.allows() is True  # one probe
        breaker.record_failure()
        assert breaker.allows() is False  # and immediately shut again


def test_a_custom_threshold_is_honoured(clock: Clock) -> None:
    breaker = CircuitBreaker(threshold=2, clock=clock)
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN


def test_breakers_are_created_per_provider_and_are_independent() -> None:
    clock = Clock()
    breakers = CircuitBreakers(clock=clock)
    for _ in range(5):
        breakers["anthropic"].record_failure()
    assert breakers.state("anthropic") is BreakerState.OPEN
    assert breakers.state("ollama") is BreakerState.CLOSED
    assert breakers["anthropic"] is breakers["anthropic"]


def test_snapshot_reports_every_seen_provider() -> None:
    breakers = CircuitBreakers(clock=Clock())
    breakers["mock"].record_success()
    for _ in range(5):
        breakers["anthropic"].record_failure()
    assert breakers.snapshot() == {
        "mock": BreakerState.CLOSED,
        "anthropic": BreakerState.OPEN,
    }


def test_reset_clears_every_breaker() -> None:
    breakers = CircuitBreakers(clock=Clock())
    for _ in range(5):
        breakers["anthropic"].record_failure()
    breakers.reset()
    assert breakers.snapshot() == {}
    assert breakers.state("anthropic") is BreakerState.CLOSED
