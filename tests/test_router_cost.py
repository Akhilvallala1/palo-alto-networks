"""Unit tests for the spend ledger and the budget policy it enforces."""

import asyncio
from datetime import date
from typing import Literal

import pytest

from conduit.config import BudgetSettings, RoutingConfig
from conduit.contracts import Complexity
from conduit.router.cost import (
    BUDGET_EXCEEDED_HEADER,
    BudgetEnforcer,
    BudgetExceededError,
    SpendLedger,
    key_fingerprint,
)
from conduit.router.policy import RoutingPolicy

KEY = "sk-test-key"


def policy() -> RoutingPolicy:
    return RoutingPolicy(
        RoutingConfig.model_validate(
            {
                "tiers": {
                    "trivial": ["cheap"],
                    "standard": ["mid"],
                    "complex": ["dear"],
                },
                "budgets": {"default_daily_usd": 5.0, "on_exceed": "downgrade_tier"},
            }
        )
    )


def enforcer(
    on_exceed: Literal["downgrade_tier", "reject"] = "downgrade_tier",
    ledger: SpendLedger | None = None,
) -> BudgetEnforcer:
    return BudgetEnforcer(
        BudgetSettings(default_daily_usd=5.0, on_exceed=on_exceed), policy(), ledger
    )


# --------------------------------------------------------------------------- #
# Ledger math
# --------------------------------------------------------------------------- #


async def test_record_accumulates_and_returns_the_running_total() -> None:
    ledger = SpendLedger()
    assert await ledger.record(KEY, 0.25) == pytest.approx(0.25)
    assert await ledger.record(KEY, 0.75) == pytest.approx(1.0)
    assert await ledger.spent(KEY) == pytest.approx(1.0)


async def test_spend_is_isolated_per_key() -> None:
    ledger = SpendLedger()
    await ledger.record(KEY, 1.0)
    await ledger.record("other-key", 2.0)
    assert await ledger.spent(KEY) == pytest.approx(1.0)
    assert await ledger.spent("other-key") == pytest.approx(2.0)


async def test_spend_resets_on_the_day_roll() -> None:
    today = date(2026, 9, 7)
    ledger = SpendLedger(clock=lambda: today)
    await ledger.record(KEY, 4.0)
    today = date(2026, 9, 8)
    assert await ledger.spent(KEY) == pytest.approx(0.0)


async def test_negative_spend_is_rejected() -> None:
    ledger = SpendLedger()
    with pytest.raises(ValueError, match="must not be negative"):
        await ledger.record(KEY, -1.0)


async def test_reset_clears_one_key_or_everything() -> None:
    ledger = SpendLedger()
    await ledger.record(KEY, 1.0)
    await ledger.record("other-key", 1.0)
    await ledger.reset(KEY)
    assert await ledger.spent(KEY) == 0.0
    assert await ledger.spent("other-key") == pytest.approx(1.0)
    await ledger.reset()
    assert await ledger.snapshot() == {}


async def test_snapshot_reports_todays_spend_only() -> None:
    today = date(2026, 9, 7)
    ledger = SpendLedger(clock=lambda: today)
    await ledger.record(KEY, 1.5)
    today = date(2026, 9, 8)
    await ledger.record(KEY, 0.5)
    assert await ledger.snapshot() == {KEY: pytest.approx(0.5)}


def test_per_key_limit_overrides_the_default() -> None:
    ledger = SpendLedger(limits={KEY: 20.0})
    assert ledger.limit_for(KEY, 5.0) == 20.0
    assert ledger.limit_for("other-key", 5.0) == 5.0
    ledger.set_limit("other-key", 1.0)
    assert ledger.limit_for("other-key", 5.0) == 1.0


def test_key_fingerprint_is_short_stable_and_not_the_key() -> None:
    fingerprint = key_fingerprint(KEY)
    assert fingerprint == key_fingerprint(KEY)
    assert len(fingerprint) == 12
    assert KEY not in fingerprint


# --------------------------------------------------------------------------- #
# Budget policy
# --------------------------------------------------------------------------- #


async def test_under_budget_passes_the_tier_through() -> None:
    decision = await enforcer().apply(KEY, Complexity.COMPLEX)
    assert decision.tier is Complexity.COMPLEX
    assert decision.exceeded is False
    assert decision.downgraded is False
    assert decision.headers() == {}
    assert decision.remaining_usd == pytest.approx(5.0)


async def test_breach_downgrades_one_tier_and_reports_it() -> None:
    guard = enforcer()
    await guard.ledger.record(KEY, 5.01)
    decision = await guard.apply(KEY, Complexity.COMPLEX)
    assert decision.tier is Complexity.STANDARD
    assert decision.requested_tier is Complexity.COMPLEX
    assert decision.exceeded is True
    assert decision.downgraded is True
    assert BUDGET_EXCEEDED_HEADER in decision.headers()
    assert "action=downgrade_tier" in decision.headers()[BUDGET_EXCEEDED_HEADER]


async def test_downgrade_floors_at_trivial() -> None:
    guard = enforcer()
    await guard.ledger.record(KEY, 99.0)
    decision = await guard.apply(KEY, Complexity.TRIVIAL)
    assert decision.tier is Complexity.TRIVIAL
    assert decision.exceeded is True
    assert decision.downgraded is False  # already at the floor, nothing to drop


async def test_reject_mode_raises_with_the_documented_header() -> None:
    guard = enforcer("reject")
    await guard.ledger.record(KEY, 5.0)
    with pytest.raises(BudgetExceededError) as caught:
        await guard.apply(KEY, Complexity.STANDARD)
    error = caught.value
    assert error.status_code == 429
    assert error.headers[BUDGET_EXCEEDED_HEADER] == (
        f"limit=5.00;spent=5.00;window={error.decision.window.isoformat()};action=reject"
    )
    assert error.key_fingerprint == key_fingerprint(KEY)
    assert KEY not in str(error)


async def test_spend_exactly_at_the_limit_is_a_breach() -> None:
    guard = enforcer()
    await guard.ledger.record(KEY, 5.0)
    decision = await guard.apply(KEY, Complexity.COMPLEX)
    assert decision.exceeded is True


async def test_unmetered_caller_is_never_budgeted() -> None:
    guard = enforcer("reject")
    decision = await guard.apply(None, Complexity.COMPLEX)
    assert decision.tier is Complexity.COMPLEX
    assert decision.exceeded is False
    assert await guard.record(None, 1.0) == 0.0


async def test_per_key_limit_is_honoured_by_the_enforcer() -> None:
    guard = enforcer(ledger=SpendLedger(limits={KEY: 0.5}))
    await guard.ledger.record(KEY, 0.6)
    decision = await guard.apply(KEY, Complexity.COMPLEX)
    assert decision.limit_usd == pytest.approx(0.5)
    assert decision.exceeded is True


# --------------------------------------------------------------------------- #
# Concurrency (issue AC-6)
# --------------------------------------------------------------------------- #


async def test_concurrent_writes_on_one_key_lose_nothing() -> None:
    ledger = SpendLedger()
    await asyncio.gather(*(ledger.record(KEY, 0.01) for _ in range(500)))
    assert await ledger.spent(KEY) == pytest.approx(5.0)


async def test_concurrent_writes_across_keys_stay_separate() -> None:
    ledger = SpendLedger()
    await asyncio.gather(
        *(ledger.record(f"key-{index % 5}", 0.10) for index in range(250)),
    )
    snapshot = await ledger.snapshot()
    assert set(snapshot) == {f"key-{index}" for index in range(5)}
    assert all(total == pytest.approx(5.0) for total in snapshot.values())
