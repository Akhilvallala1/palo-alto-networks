"""Ordered failover across a tier's model chain, with a circuit breaker.

Retry (`base.call_with_retry`) and failover are deliberately different layers.
Retry asks "would the *same* provider work if I asked again?" — so it only
fires on 5xx, timeouts and 429. Failover asks "is there *another* provider?" —
so it fires on anything, including a 401, because the next hop has different
credentials and a different blast radius.

The breaker sits between them. Once a provider has failed five times in a row
it is skipped without a call at all, which is what stops a dead vendor from
adding its full retry budget to the latency of every request. After 30s one
probe is allowed through; success closes the breaker, failure reopens it for
another 30s.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from conduit.contracts import CompletionRequest, CompletionResponse, Complexity

from .base import ProviderError, ProviderUnavailableError
from .registry import ModelRegistry

__all__ = [
    "FAILURE_THRESHOLD",
    "RECOVERY_SECONDS",
    "AllProvidersFailedError",
    "BreakerState",
    "CircuitBreaker",
    "CircuitBreakers",
    "CircuitOpenError",
    "FailoverChain",
    "FailoverEvent",
]

#: Consecutive failures that open a provider's breaker.
FAILURE_THRESHOLD = 5

#: How long a breaker stays hard-open before it allows one probe.
RECOVERY_SECONDS = 30.0


class BreakerState(str, Enum):
    CLOSED = "closed"  # healthy; calls pass through
    OPEN = "open"  # failing; calls are rejected without being made
    HALF_OPEN = "half_open"  # recovery window elapsed; one probe allowed


class CircuitOpenError(ProviderError):
    """The breaker rejected the call without making it. Never retryable."""


class AllProvidersFailedError(ProviderError):
    """Every model in the chain failed or was short-circuited."""

    def __init__(self, attempted: Sequence[str], last_error: BaseException | None) -> None:
        tried = ", ".join(attempted) if attempted else "no usable models"
        super().__init__("failover", f"exhausted chain [{tried}]: {last_error}")
        self.attempted = tuple(attempted)
        self.last_error = last_error


# --------------------------------------------------------------------------- #
# Breaker
# --------------------------------------------------------------------------- #


@dataclass
class CircuitBreaker:
    """Per-provider breaker. `clock` is injectable so 30s costs a test nothing.

    State is derived from `opened_at` rather than stored, which removes the
    class of bug where a breaker is left in HALF_OPEN because nothing happened
    to transition it out.
    """

    threshold: int = FAILURE_THRESHOLD
    recovery_s: float = RECOVERY_SECONDS
    clock: Callable[[], float] = time.monotonic
    consecutive_failures: int = 0
    opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        if self.opened_at is None:
            return BreakerState.CLOSED
        if self.clock() - self.opened_at >= self.recovery_s:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allows(self) -> bool:
        """True when a call may be made — closed, or half-open for one probe."""
        return self.state is not BreakerState.OPEN

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        was_half_open = self.state is BreakerState.HALF_OPEN
        self.consecutive_failures += 1
        if was_half_open or self.consecutive_failures >= self.threshold:
            # A failed probe reopens for a fresh window rather than letting the
            # next call through immediately.
            self.opened_at = self.clock()


class CircuitBreakers:
    """One breaker per provider name, created on first sight."""

    def __init__(
        self,
        *,
        threshold: int = FAILURE_THRESHOLD,
        recovery_s: float = RECOVERY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._recovery_s = recovery_s
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}

    def __getitem__(self, provider_name: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider_name)
        if breaker is None:
            breaker = CircuitBreaker(
                threshold=self._threshold, recovery_s=self._recovery_s, clock=self._clock
            )
            self._breakers[provider_name] = breaker
        return breaker

    def state(self, provider_name: str) -> BreakerState:
        return self[provider_name].state

    def snapshot(self) -> dict[str, BreakerState]:
        """Current state per provider, for telemetry and `/metrics`."""
        return {name: breaker.state for name, breaker in self._breakers.items()}

    def reset(self) -> None:
        self._breakers.clear()


# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FailoverEvent:
    """One hop's outcome, for whoever is doing the logging.

    Emitting a struct rather than logging directly keeps `structlog` out of this
    package and lets the telemetry component decide what a circuit-breaker event
    looks like on the wire.
    """

    model: str
    provider: str
    outcome: Literal["success", "failure", "circuit_open"]
    breaker_state: BreakerState
    error: str | None = None


@dataclass
class _Attempt:
    attempted: list[str] = field(default_factory=list)
    last_error: BaseException | None = None


class FailoverChain:
    """Walks a tier's ordered model chain until one provider answers."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        breakers: CircuitBreakers | None = None,
        on_event: Callable[[FailoverEvent], None] | None = None,
    ) -> None:
        self.registry = registry
        self.breakers = breakers if breakers is not None else CircuitBreakers()
        self._on_event = on_event

    async def complete(
        self,
        req: CompletionRequest,
        chain: Sequence[str],
        tier: Complexity,
    ) -> CompletionResponse:
        """Try each model in order; return the first success.

        The returned response carries the caller's `tier` — the chain knows the
        routing decision and the provider does not — and a `fallback_from`
        listing every model that was tried and did not answer, in order,
        including ones skipped by an open breaker. Raises `AllProvidersFailedError`
        when nothing in the chain answers.

        Chain entries whose provider is dormant are dropped rather than
        attempted; tier eligibility is left to the router, which built the
        chain, so this layer never second-guesses a deliberate routing choice.
        """
        run = _Attempt()
        for model_id in self.registry.usable_chain(chain):
            response = await self._try(req, model_id, run)
            if response is not None:
                return response.model_copy(
                    update={"routed_tier": tier, "fallback_from": list(run.attempted)}
                )
        raise AllProvidersFailedError(run.attempted, run.last_error)

    async def _try(
        self, req: CompletionRequest, model_id: str, run: _Attempt
    ) -> CompletionResponse | None:
        provider = self.registry.provider_for(model_id)
        breaker = self.breakers[provider.name]

        if not breaker.allows():
            error = CircuitOpenError(
                provider.name, f"circuit open, skipped {model_id} without calling"
            )
            run.attempted.append(model_id)
            run.last_error = error
            self._emit(model_id, provider.name, "circuit_open", breaker.state, str(error))
            return None

        try:
            response = await provider.complete(req, model_id)
        except ProviderError as exc:
            self._record_failure(model_id, provider.name, run, exc)
            return None
        except Exception as exc:  # an adapter that broke its own contract
            wrapped = ProviderUnavailableError(provider.name, f"unexpected error: {exc!r}")
            wrapped.__cause__ = exc
            self._record_failure(model_id, provider.name, run, wrapped)
            return None

        breaker.record_success()
        self._emit(model_id, provider.name, "success", breaker.state)
        return response

    def _record_failure(
        self, model_id: str, provider_name: str, run: _Attempt, error: ProviderError
    ) -> None:
        breaker = self.breakers[provider_name]
        breaker.record_failure()
        run.attempted.append(model_id)
        run.last_error = error
        self._emit(model_id, provider_name, "failure", breaker.state, str(error))

    def _emit(
        self,
        model: str,
        provider: str,
        outcome: Literal["success", "failure", "circuit_open"],
        state: BreakerState,
        error: str | None = None,
    ) -> None:
        if self._on_event is not None:
            self._on_event(FailoverEvent(model, provider, outcome, state, error))
