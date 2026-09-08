"""Shared machinery for every provider adapter.

Error classification, retry policy, timeout handling and usage pricing live
here so the vendor adapters stay thin. An adapter's only job is to shape a
request for its API, parse the reply, and hand back a `RawCompletion`;
`BaseProvider` turns that into the `CompletionResponse` the frozen contract
requires, complete with a measured latency and a priced `Usage`.

Two rules from the epic are enforced in this module rather than per adapter:

- Retries are 3 attempts, exponential backoff with jitter, capped at 8s, and a
  4xx is never retried except 429.
- Prices come from `config/models.yaml` via `ModelSpec`. Nothing here hardcodes
  a dollar figure; the only constants are the cache multipliers, which are
  ratios fixed by `contracts.Usage`, not prices.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from conduit.config import ModelSpec
from conduit.contracts import (
    CompletionRequest,
    CompletionResponse,
    Complexity,
    Message,
    Usage,
)

__all__ = [
    "CACHE_READ_MULTIPLIER",
    "CACHE_WRITE_MULTIPLIER",
    "TIER_METADATA_KEY",
    "BaseProvider",
    "HttpProvider",
    "ProviderError",
    "ProviderRefusedError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RateLimitedError",
    "RawCompletion",
    "RetryPolicy",
    "Sleeper",
    "as_int",
    "as_mapping",
    "as_str",
    "call_with_retry",
    "classify_http_error",
    "price_usage",
    "resolve_tier",
    "split_system",
]

# `Usage` documents cache reads as 0.1x input and 5m cache writes as 1.25x
# input. Those are billing ratios set by the contract, not prices, so they are
# the one thing in this module that is allowed to be a literal.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

#: `CompletionRequest.metadata` key the router uses to tell a provider which
#: tier it resolved. Providers cannot classify complexity themselves.
TIER_METADATA_KEY = "tier"

T = TypeVar("T")

Sleeper = Callable[[float], Awaitable[None]]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ProviderError(Exception):
    """A provider call failed.

    `retryable` says whether hitting the *same* provider again could plausibly
    work. It is deliberately separate from the failover decision: a 401 is not
    retryable but is still worth failing over from, because the next provider
    in the chain has different credentials.
    """

    retryable = False

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status_code = status_code


class ProviderTimeoutError(ProviderError):
    """The provider did not answer inside the configured timeout."""

    retryable = True


class ProviderUnavailableError(ProviderError):
    """A 5xx or a transport-level failure: the provider is up but not working."""

    retryable = True


class RateLimitedError(ProviderError):
    """HTTP 429. The one 4xx worth retrying."""

    retryable = True

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = 429,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(provider, message, status_code=status_code)
        self.retry_after_s = retry_after_s


class ProviderRefusedError(ProviderError):
    """A 4xx other than 429: bad request, bad key, forbidden. Never retried."""

    retryable = False


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:  # HTTP-date form; backoff handles it well enough
        return None


def classify_http_error(provider: str, response: httpx.Response) -> ProviderError:
    """Map an HTTP status onto the retry policy the epic specifies."""
    status = response.status_code
    detail = response.text.strip()[:400] or response.reason_phrase
    if status == 429:
        return RateLimitedError(
            provider,
            f"rate limited: {detail}",
            status_code=status,
            retry_after_s=_retry_after(response),
        )
    if 400 <= status < 500:
        return ProviderRefusedError(provider, f"HTTP {status}: {detail}", status_code=status)
    return ProviderUnavailableError(provider, f"HTTP {status}: {detail}", status_code=status)


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetryPolicy:
    """3 attempts, exponential backoff with jitter, capped at 8s.

    Jitter is the equal-jitter variant: half the computed backoff plus a random
    share of the other half. That keeps every delay strictly positive and
    strictly under the cap, which a full-jitter policy does not, and it still
    spreads a thundering herd.
    """

    attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def backoff_s(self, attempt: int, rand: float) -> float:
        """Delay before the attempt after `attempt` (1-based) has failed."""
        window = min(self.base_delay_s * (1 << max(0, attempt - 1)), self.max_delay_s)
        return window / 2 + window / 2 * rand


async def call_with_retry(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Sleeper = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Run `operation`, retrying only errors that mark themselves retryable.

    `sleep` and `rand` are injectable so the backoff schedule can be asserted
    in a test without spending eight real seconds on it.
    """
    if policy.attempts < 1:
        raise ValueError("RetryPolicy.attempts must be >= 1")
    last: ProviderError | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return await operation()
        except ProviderError as exc:
            last = exc
            if not exc.retryable or attempt == policy.attempts:
                raise
            await sleep(policy.backoff_s(attempt, rand()))
    raise last if last is not None else RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


def price_usage(
    spec: ModelSpec,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Cost in USD for one call, using only prices from `config/models.yaml`."""
    input_rate = spec.input_per_1m_usd / 1_000_000
    output_rate = spec.output_per_1m_usd / 1_000_000
    return (
        prompt_tokens * input_rate
        + cache_read_tokens * input_rate * CACHE_READ_MULTIPLIER
        + cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        + completion_tokens * output_rate
    )


def resolve_tier(req: CompletionRequest, spec: ModelSpec) -> Complexity:
    """Best available answer to a question providers cannot actually answer.

    `CompletionResponse.routed_tier` is required by the frozen contract, but
    complexity classification belongs to the router, which is a different
    component. The router therefore passes its verdict down in
    `metadata["tier"]`; absent that, we fall back to the cheapest tier this
    model is eligible for, which is deterministic and never lies upward. The
    failover layer overwrites this with the real tier when it has one.
    """
    hint = req.metadata.get(TIER_METADATA_KEY)
    if hint is not None:
        try:
            return Complexity(hint)
        except ValueError:
            pass
    eligible = set(spec.tiers)
    for tier in Complexity:
        if tier in eligible:
            return tier
    return Complexity.STANDARD


# --------------------------------------------------------------------------- #
# Base adapter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RawCompletion:
    """What an adapter parses out of a vendor reply, before pricing."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class BaseProvider(ABC):
    """Retry, timeout, timing and pricing, shared by every adapter.

    Subclasses implement `_invoke` (one attempt against the vendor) and
    `_probe` (a cheap reachability check). Everything the `Provider` protocol
    promises — a `provider` equal to `self.name`, a concrete `model`, a priced
    `usage`, a measured `latency_ms`, an empty `fallback_from`, a `health` that
    never raises — is upheld here so no adapter can forget it.
    """

    name: str = "base"

    def __init__(
        self,
        *,
        models: Mapping[str, ModelSpec],
        timeout_s: float = 30.0,
        retry: RetryPolicy | None = None,
        sleep: Sleeper = asyncio.sleep,
        rand: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._models = dict(models)
        self._timeout_s = timeout_s
        self._retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep
        self._rand = rand
        self._clock = clock

    # -- Provider protocol -------------------------------------------------- #

    def supports(self, model: str) -> bool:
        """Pure: exactly the model ids this adapter was configured to serve."""
        return model in self._models

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        spec = self._spec(model)
        started = self._clock()
        raw = await call_with_retry(
            lambda: self._attempt(req, model),
            self._retry,
            sleep=self._sleep,
            rand=self._rand,
        )
        latency_ms = max(0, int((self._clock() - started) * 1000))
        usage = Usage(
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            cost_usd=price_usage(
                spec,
                prompt_tokens=raw.prompt_tokens,
                completion_tokens=raw.completion_tokens,
                cache_read_tokens=raw.cache_read_tokens,
                cache_write_tokens=raw.cache_write_tokens,
            ),
            cache_read_tokens=raw.cache_read_tokens,
            cache_write_tokens=raw.cache_write_tokens,
        )
        return CompletionResponse(
            text=raw.text,
            model=model,
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            routed_tier=resolve_tier(req, spec),
        )

    async def health(self) -> bool:
        """Reachability only. Never raises; any error is a False."""
        try:
            return await asyncio.wait_for(self._probe(), self._timeout_s)
        except Exception:
            return False

    # -- Subclass surface --------------------------------------------------- #

    @abstractmethod
    async def _invoke(self, req: CompletionRequest, model: str) -> RawCompletion:
        """One attempt against the backend. Raise `ProviderError` on failure."""

    @abstractmethod
    async def _probe(self) -> bool:
        """One cheap reachability check. May raise; `health` swallows it."""

    async def aclose(self) -> None:  # noqa: B027 - concrete no-op, not an abstract hook
        """Release any transport the adapter holds. Idempotent.

        Deliberately not abstract: only the HTTP adapters hold anything to
        release, and forcing `mock` to declare an empty override would be noise.
        """

    # -- Internals ---------------------------------------------------------- #

    def _spec(self, model: str) -> ModelSpec:
        spec = self._models.get(model)
        if spec is None:
            raise ProviderRefusedError(self.name, f"unsupported model {model!r}")
        return spec

    async def _attempt(self, req: CompletionRequest, model: str) -> RawCompletion:
        try:
            return await asyncio.wait_for(self._invoke(req, model), self._timeout_s)
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                self.name, f"no response for {model!r} within {self._timeout_s}s"
            ) from exc


# --------------------------------------------------------------------------- #
# Reply parsing
# --------------------------------------------------------------------------- #
#
# Vendor JSON is untyped at the boundary. These narrow it once, so adapters read
# as declarative field lists rather than isinstance ladders, and a vendor that
# omits a token count yields 0 rather than a TypeError three frames later.


def as_mapping(value: object) -> dict[str, Any]:
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Peel system turns off the front, for APIs that take them out of band."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    turns = [m for m in messages if m.role != "system"]
    return (system or None), turns


class HttpProvider(BaseProvider):
    """A `BaseProvider` that talks JSON over HTTP.

    The transport is injectable, which is the whole testing strategy for the
    live adapters: `httpx.MockTransport` gives real request shaping and real
    status-code handling with no socket, so CI never touches the network.
    """

    def __init__(
        self,
        *,
        base_url: str,
        headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=dict(headers or {}),
            transport=transport,
            timeout=self._timeout_s,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=json, params=params)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"{method} {path} timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(self.name, f"{method} {path} failed: {exc}") from exc
        if response.status_code >= 400:
            raise classify_http_error(self.name, response)
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(self.name, f"{method} {path} returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderUnavailableError(
                self.name, f"{method} {path} returned {type(payload).__name__}, expected object"
            )
        return {str(key): value for key, value in payload.items()}

    async def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self._request("POST", path, json=payload)

    async def _get(self, path: str, *, params: Mapping[str, str] | None = None) -> dict[str, Any]:
        return await self._request("GET", path, params=params)
