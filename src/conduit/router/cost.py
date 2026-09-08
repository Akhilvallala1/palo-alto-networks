"""Per-API-key daily spend ledger and the budget policy it enforces.

The ledger is an in-memory, UTC-day-bucketed accumulator guarded by an
`asyncio.Lock`, so concurrent requests on the same key cannot lose a write
(epic AC-11, issue AC-6). It stores no key material: every key is reduced to a
short salted-free SHA-256 prefix for logging, and the raw key is only ever used
as a dict lookup.

On breach, `config/routing.yaml`'s `budgets.on_exceed` decides:

* `downgrade_tier` — serve the request one tier cheaper (floored at trivial)
  and mark the decision, so the caller still gets a 200.
* `reject` — raise `BudgetExceededError`, which the gateway renders as HTTP 429 with
  the `X-Conduit-Budget-Exceeded` header.

Header format (documented so the gateway and its tests agree)::

    X-Conduit-Budget-Exceeded: limit=5.00;spent=5.40;window=2026-09-07;action=reject
"""

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, date, datetime

from pydantic import BaseModel

from conduit.config import BudgetSettings
from conduit.contracts import Complexity

from .policy import RoutingPolicy

__all__ = [
    "BUDGET_EXCEEDED_HEADER",
    "BudgetDecision",
    "BudgetEnforcer",
    "BudgetExceededError",
    "SpendLedger",
]

BUDGET_EXCEEDED_HEADER = "X-Conduit-Budget-Exceeded"
BUDGET_EXCEEDED_STATUS = 429


def key_fingerprint(api_key: str) -> str:
    """A short, stable, non-reversible id for logs. Never log the key itself."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def _utc_today() -> date:
    return datetime.now(UTC).date()


class BudgetDecision(BaseModel):
    """What the budget engine did to a request, and why."""

    tier: Complexity
    requested_tier: Complexity
    spent_usd: float
    limit_usd: float
    exceeded: bool = False
    downgraded: bool = False
    window: date

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def header_value(self, action: str = "downgrade_tier") -> str:
        return (
            f"limit={self.limit_usd:.2f};spent={self.spent_usd:.2f};"
            f"window={self.window.isoformat()};action={action}"
        )

    def headers(self) -> dict[str, str]:
        """Response headers to attach. Empty unless the budget was breached."""
        if not self.exceeded:
            return {}
        action = "downgrade_tier" if self.downgraded else "observed"
        return {BUDGET_EXCEEDED_HEADER: self.header_value(action)}


class BudgetExceededError(Exception):
    """Raised under `on_exceed: reject`. Carries its own HTTP rendering."""

    status_code = BUDGET_EXCEEDED_STATUS

    def __init__(self, decision: BudgetDecision, api_key: str = "") -> None:
        self.decision = decision
        self.key_fingerprint = key_fingerprint(api_key) if api_key else ""
        super().__init__(
            f"daily budget exceeded: spent ${decision.spent_usd:.4f} of "
            f"${decision.limit_usd:.2f} on {decision.window.isoformat()}"
        )

    @property
    def headers(self) -> dict[str, str]:
        return {BUDGET_EXCEEDED_HEADER: self.decision.header_value("reject")}


class SpendLedger:
    """Daily spend per API key. Concurrency-safe; resets on the UTC day roll."""

    def __init__(
        self,
        *,
        clock: Callable[[], date] = _utc_today,
        limits: dict[str, float] | None = None,
    ) -> None:
        self._clock = clock
        self._limits = dict(limits or {})
        self._spend: dict[tuple[date, str], float] = {}
        self._lock = asyncio.Lock()

    @property
    def today(self) -> date:
        return self._clock()

    async def record(self, api_key: str, cost_usd: float) -> float:
        """Add spend and return the key's new running total for the day."""
        if cost_usd < 0:
            raise ValueError("cost_usd must not be negative")
        async with self._lock:
            bucket = (self._clock(), api_key)
            total = self._spend.get(bucket, 0.0) + cost_usd
            self._spend[bucket] = total
            return total

    async def spent(self, api_key: str) -> float:
        async with self._lock:
            return self._spend.get((self._clock(), api_key), 0.0)

    def limit_for(self, api_key: str, default_usd: float) -> float:
        """Per-key override if one is configured, else the default."""
        return self._limits.get(api_key, default_usd)

    def set_limit(self, api_key: str, limit_usd: float) -> None:
        self._limits[api_key] = limit_usd

    async def reset(self, api_key: str | None = None) -> None:
        async with self._lock:
            if api_key is None:
                self._spend.clear()
                return
            today = self._clock()
            self._spend.pop((today, api_key), None)

    async def snapshot(self) -> dict[str, float]:
        """Today's spend per key — for telemetry and the benchmark script."""
        async with self._lock:
            today = self._clock()
            return {key: usd for (day, key), usd in self._spend.items() if day == today}


class BudgetEnforcer:
    """Applies `budgets.on_exceed` to a routing decision."""

    def __init__(
        self,
        budgets: BudgetSettings,
        policy: RoutingPolicy,
        ledger: SpendLedger | None = None,
    ) -> None:
        self.budgets = budgets
        self.policy = policy
        self.ledger = ledger or SpendLedger()

    async def apply(self, api_key: str | None, tier: Complexity) -> BudgetDecision:
        """Check the key's day spend and adjust — or refuse — the tier.

        An unauthenticated / unmetered caller (`api_key=None`) is passed
        through untouched: budgets are a per-key concept.
        """
        window = self.ledger.today
        if api_key is None:
            return BudgetDecision(
                tier=tier,
                requested_tier=tier,
                spent_usd=0.0,
                limit_usd=self.budgets.default_daily_usd,
                window=window,
            )

        spent = await self.ledger.spent(api_key)
        limit = self.ledger.limit_for(api_key, self.budgets.default_daily_usd)
        decision = BudgetDecision(
            tier=tier,
            requested_tier=tier,
            spent_usd=spent,
            limit_usd=limit,
            window=window,
        )
        if spent < limit:
            return decision

        decision.exceeded = True
        if self.budgets.on_exceed == "reject":
            raise BudgetExceededError(decision, api_key)

        decision.tier = self.policy.downgrade(tier)
        decision.downgraded = decision.tier != tier
        return decision

    async def record(self, api_key: str | None, cost_usd: float) -> float:
        """Post-flight accounting. No-op for unmetered callers."""
        if api_key is None:
            return 0.0
        return await self.ledger.record(api_key, cost_usd)
