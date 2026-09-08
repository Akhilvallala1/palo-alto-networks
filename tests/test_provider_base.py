"""Unit tests for the shared provider machinery: pricing, retry, classification.

Cost math is checked against hand-computed figures derived from
`config/models.yaml`, not against constants in the code, so a price edit that
the logic silently ignores fails here.
"""

import asyncio

import httpx
import pytest

from conduit.config import ModelSpec
from conduit.contracts import CompletionRequest, Complexity, Message
from conduit.providers.base import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    BaseProvider,
    ProviderError,
    ProviderRefusedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitedError,
    RawCompletion,
    RetryPolicy,
    call_with_retry,
    classify_http_error,
    price_usage,
    resolve_tier,
    split_system,
)
from provider_fixtures import ECHO, HAIKU, LLAMA, OPUS, SONNET, no_sleep, repo_models, request

MODELS = repo_models().models


# --------------------------------------------------------------------------- #
# Cost math (AC-2)
# --------------------------------------------------------------------------- #


def test_cost_matches_a_hand_computed_price_from_the_yaml() -> None:
    """Haiku is $1.00/1M in, $5.00/1M out. 1M in + 200k out = 1.00 + 1.00."""
    spec = MODELS[HAIKU]
    assert spec.input_per_1m_usd == 1.00
    assert spec.output_per_1m_usd == 5.00
    cost = price_usage(spec, prompt_tokens=1_000_000, completion_tokens=200_000)
    assert cost == pytest.approx(2.00)


@pytest.mark.parametrize(
    ("model", "prompt", "completion", "expected"),
    [
        (OPUS, 1_000_000, 1_000_000, 5.00 + 25.00),
        (SONNET, 500_000, 100_000, 1.00 + 1.00),
        (HAIKU, 120, 40, 120 * 1e-6 + 40 * 5e-6),
        (LLAMA, 10_000, 10_000, 0.0),
        (ECHO, 10_000, 10_000, 0.0),
    ],
)
def test_cost_for_each_registered_model(
    model: str, prompt: int, completion: int, expected: float
) -> None:
    cost = price_usage(MODELS[model], prompt_tokens=prompt, completion_tokens=completion)
    assert cost == pytest.approx(expected)


def test_cache_reads_are_priced_at_a_tenth_of_input() -> None:
    spec = MODELS[SONNET]
    cost = price_usage(spec, prompt_tokens=0, completion_tokens=0, cache_read_tokens=1_000_000)
    assert cost == pytest.approx(spec.input_per_1m_usd * CACHE_READ_MULTIPLIER)
    assert cost == pytest.approx(0.20)


def test_cache_writes_are_priced_at_1_25x_input() -> None:
    spec = MODELS[SONNET]
    cost = price_usage(spec, prompt_tokens=0, completion_tokens=0, cache_write_tokens=1_000_000)
    assert cost == pytest.approx(spec.input_per_1m_usd * CACHE_WRITE_MULTIPLIER)
    assert cost == pytest.approx(2.50)


def test_cost_sums_every_token_class() -> None:
    spec = MODELS[OPUS]
    cost = price_usage(
        spec,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost == pytest.approx(5.00 + 25.00 + 0.50 + 6.25)


def test_zero_tokens_cost_nothing() -> None:
    assert price_usage(MODELS[OPUS], prompt_tokens=0, completion_tokens=0) == 0.0


def test_pricing_reads_the_spec_rather_than_a_hardcoded_table() -> None:
    """A price only present in a synthetic spec must still be honoured."""
    spec = ModelSpec(
        provider="anthropic",
        input_per_1m_usd=3.0,
        output_per_1m_usd=7.0,
        context_window=1000,
        tiers=[Complexity.STANDARD],
    )
    assert price_usage(spec, prompt_tokens=1_000_000, completion_tokens=0) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


def test_backoff_grows_exponentially() -> None:
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=8.0)
    #  rand=1.0 gives the top of each jitter window: 0.5, 1.0, 2.0, 4.0, 8.0
    assert [policy.backoff_s(n, 1.0) for n in range(1, 6)] == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_backoff_is_capped_at_eight_seconds() -> None:
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=8.0)
    assert all(policy.backoff_s(n, 1.0) <= 8.0 for n in range(1, 25))
    assert policy.backoff_s(20, 1.0) == 8.0


def test_backoff_is_jittered_inside_the_window() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=8.0)
    low, high = policy.backoff_s(3, 0.0), policy.backoff_s(3, 1.0)
    assert low == 2.0 and high == 4.0
    assert low < policy.backoff_s(3, 0.5) < high


def test_backoff_is_always_positive() -> None:
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=8.0)
    assert all(policy.backoff_s(n, 0.0) > 0 for n in range(1, 6))


class _Flaky:
    """Fails with `error` for the first `failures` calls, then succeeds."""

    def __init__(self, error: ProviderError, failures: int) -> None:
        self.error = error
        self.failures = failures
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


async def test_retry_succeeds_on_a_later_attempt() -> None:
    op = _Flaky(ProviderUnavailableError("x", "500"), failures=2)
    assert await call_with_retry(op, RetryPolicy(), sleep=no_sleep) == "ok"
    assert op.calls == 3


async def test_retry_makes_exactly_three_attempts_before_giving_up() -> None:
    op = _Flaky(ProviderUnavailableError("x", "500"), failures=99)
    with pytest.raises(ProviderUnavailableError):
        await call_with_retry(op, RetryPolicy(), sleep=no_sleep)
    assert op.calls == 3


async def test_a_rate_limit_is_retried() -> None:
    """AC-6: 429 is the one 4xx worth asking again about."""
    op = _Flaky(RateLimitedError("x", "429"), failures=1)
    assert await call_with_retry(op, RetryPolicy(), sleep=no_sleep) == "ok"
    assert op.calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_a_refusal_is_never_retried(status: int) -> None:
    """AC-6: a 4xx that is not 429 fails on the first attempt."""
    op = _Flaky(ProviderRefusedError("x", str(status), status_code=status), failures=99)
    with pytest.raises(ProviderRefusedError):
        await call_with_retry(op, RetryPolicy(), sleep=no_sleep)
    assert op.calls == 1


async def test_a_timeout_is_retried() -> None:
    op = _Flaky(ProviderTimeoutError("x", "slow"), failures=2)
    assert await call_with_retry(op, RetryPolicy(), sleep=no_sleep) == "ok"
    assert op.calls == 3


async def test_retry_sleeps_between_attempts_and_not_after_the_last() -> None:
    delays: list[float] = []

    async def record(seconds: float) -> None:
        delays.append(seconds)

    op = _Flaky(ProviderUnavailableError("x", "500"), failures=99)
    with pytest.raises(ProviderUnavailableError):
        await call_with_retry(op, RetryPolicy(), sleep=record, rand=lambda: 1.0)
    assert delays == [0.5, 1.0]  # two gaps for three attempts


async def test_retry_rejects_a_policy_with_no_attempts() -> None:
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        await call_with_retry(
            _Flaky(ProviderUnavailableError("x", "!"), 0), RetryPolicy(attempts=0)
        )


async def test_a_single_attempt_policy_does_not_retry() -> None:
    op = _Flaky(ProviderUnavailableError("x", "500"), failures=99)
    with pytest.raises(ProviderUnavailableError):
        await call_with_retry(op, RetryPolicy(attempts=1), sleep=no_sleep)
    assert op.calls == 1


# --------------------------------------------------------------------------- #
# HTTP error classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (400, ProviderRefusedError, False),
        (401, ProviderRefusedError, False),
        (403, ProviderRefusedError, False),
        (404, ProviderRefusedError, False),
        (429, RateLimitedError, True),
        (500, ProviderUnavailableError, True),
        (502, ProviderUnavailableError, True),
        (503, ProviderUnavailableError, True),
    ],
)
def test_status_codes_map_to_the_documented_retry_behaviour(
    status: int, expected: type[ProviderError], retryable: bool
) -> None:
    error = classify_http_error("anthropic", httpx.Response(status, text="nope"))
    assert isinstance(error, expected)
    assert error.retryable is retryable
    assert error.status_code == status
    assert error.provider == "anthropic"


def test_rate_limit_captures_a_numeric_retry_after() -> None:
    error = classify_http_error("x", httpx.Response(429, headers={"retry-after": "12"}))
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_s == 12.0


def test_rate_limit_tolerates_an_http_date_retry_after() -> None:
    header = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    error = classify_http_error("x", httpx.Response(429, headers=header))
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_s is None


# --------------------------------------------------------------------------- #
# Tier resolution and message shaping
# --------------------------------------------------------------------------- #


def test_resolve_tier_prefers_the_routers_hint() -> None:
    req = request(metadata={"tier": "complex"})
    assert resolve_tier(req, MODELS[HAIKU]) is Complexity.COMPLEX


def test_resolve_tier_falls_back_to_the_cheapest_eligible_tier() -> None:
    assert resolve_tier(request(), MODELS[HAIKU]) is Complexity.TRIVIAL
    assert resolve_tier(request(), MODELS[SONNET]) is Complexity.STANDARD
    assert resolve_tier(request(), MODELS[OPUS]) is Complexity.COMPLEX


def test_resolve_tier_ignores_an_unparseable_hint() -> None:
    req = request(metadata={"tier": "extremely-hard"})
    assert resolve_tier(req, MODELS[OPUS]) is Complexity.COMPLEX


def test_split_system_moves_system_turns_out_of_band() -> None:
    messages = [
        Message(role="system", content="You are terse."),
        Message(role="user", content="hi"),
        Message(role="system", content="Cite policy."),
    ]
    system, turns = split_system(messages)
    assert system == "You are terse.\n\nCite policy."
    assert [m.role for m in turns] == ["user"]


def test_split_system_returns_none_when_there_is_no_system_turn() -> None:
    system, turns = split_system([Message(role="user", content="hi")])
    assert system is None
    assert len(turns) == 1


# --------------------------------------------------------------------------- #
# BaseProvider plumbing
# --------------------------------------------------------------------------- #


class _SlowProvider(BaseProvider):
    name = "slow"

    async def _invoke(self, req: CompletionRequest, model: str) -> RawCompletion:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    async def _probe(self) -> bool:
        await asyncio.sleep(10)
        return True


async def test_a_hung_call_becomes_a_retryable_provider_timeout() -> None:
    provider = _SlowProvider(
        models={ECHO: MODELS[ECHO]}, timeout_s=0.01, retry=RetryPolicy(attempts=1)
    )
    with pytest.raises(ProviderTimeoutError) as caught:
        await provider.complete(request(), ECHO)
    assert caught.value.retryable is True


async def test_health_returns_false_instead_of_raising_on_timeout() -> None:
    provider = _SlowProvider(models={ECHO: MODELS[ECHO]}, timeout_s=0.01)
    assert await provider.health() is False


async def test_an_unsupported_model_is_refused_without_a_call() -> None:
    provider = _SlowProvider(models={ECHO: MODELS[ECHO]}, timeout_s=0.01)
    assert provider.supports(ECHO) is True
    assert provider.supports(OPUS) is False
    with pytest.raises(ProviderRefusedError, match="unsupported model"):
        await provider.complete(request(), OPUS)
