"""Integration tests: the router behind a real HTTP surface.

The gateway (#6) is not built yet, so these tests stand up a minimal FastAPI
app that consumes the router exactly as the gateway is meant to — route,
call a provider, record spend. That is enough to assert the acceptance criteria
in their stated terms: a 200 at a lower tier under `downgrade_tier`, and a 429
carrying `X-Conduit-Budget-Exceeded` under `reject`.

No network: the app is driven over an ASGI transport and the provider is a fake.
"""

import asyncio
from typing import Literal

import httpx
import pytest
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse

from conduit.config import ConduitConfig, GuardsConfig, ModelsConfig, ModelSpec, RoutingConfig
from conduit.contracts import CompletionRequest, Complexity
from conduit.router import BUDGET_EXCEEDED_HEADER, BudgetExceededError, Router, SpendLedger
from router_fakes import FakeProvider

COST_PER_CALL = 0.60

MODELS = ModelsConfig(
    models={
        "cheap": ModelSpec(
            provider="fake",
            input_per_1m_usd=1.0,
            output_per_1m_usd=5.0,
            context_window=100_000,
            tiers=[Complexity.TRIVIAL],
        ),
        "mid": ModelSpec(
            provider="fake",
            input_per_1m_usd=2.0,
            output_per_1m_usd=10.0,
            context_window=100_000,
            tiers=[Complexity.STANDARD],
        ),
        "dear": ModelSpec(
            provider="fake",
            input_per_1m_usd=5.0,
            output_per_1m_usd=25.0,
            context_window=100_000,
            tiers=[Complexity.COMPLEX],
        ),
    }
)


def build_config(on_exceed: Literal["downgrade_tier", "reject"]) -> ConduitConfig:
    return ConduitConfig(
        routing=RoutingConfig.model_validate(
            {
                "tiers": {"trivial": ["cheap"], "standard": ["mid"], "complex": ["dear"]},
                "budgets": {"default_daily_usd": 1.0, "on_exceed": on_exceed},
            }
        ),
        models=MODELS,
        guards=GuardsConfig(),
    )


def build_app(router: Router) -> FastAPI:
    """The smallest gateway that exercises the router's contract."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def complete(
        payload: CompletionRequest, x_api_key: str = Header(default="")
    ) -> JSONResponse:
        try:
            route = await router.route(payload, api_key=x_api_key)
        except BudgetExceededError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": "budget_exceeded", "detail": str(exc)},
                headers=exc.headers,
            )
        # A provider call would happen here; charge a flat rate for the test.
        spent = await router.record_spend(x_api_key, COST_PER_CALL)
        return JSONResponse(
            status_code=200,
            content={"tier": route.tier.value, "model": route.primary, "spent": spent},
            headers=route.headers(),
        )

    return app


def client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router.test")


COMPLEX_PROMPT = (
    "Analyze this discount request against the policy hierarchy and explain "
    "which clause drove the decision. Request: 26% off list on a two-year term."
)


def body(text: str = COMPLEX_PROMPT) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": text}]}


async def test_budget_breach_downgrades_and_still_returns_200() -> None:
    router = Router(build_config("downgrade_tier"))
    app = build_app(router)
    async with client(app) as http:
        first = await http.post("/v1/chat/completions", json=body(), headers={"x-api-key": "k1"})
        assert first.status_code == 200
        assert first.json()["tier"] == "complex"
        assert first.json()["model"] == "dear"
        assert BUDGET_EXCEEDED_HEADER not in first.headers

        # Two calls at $0.60 put the key over its $1.00 daily limit.
        await http.post("/v1/chat/completions", json=body(), headers={"x-api-key": "k1"})
        breached = await http.post("/v1/chat/completions", json=body(), headers={"x-api-key": "k1"})

    assert breached.status_code == 200
    assert breached.json()["tier"] == "standard"  # one rung down from complex
    assert breached.json()["model"] == "mid"
    assert "action=downgrade_tier" in breached.headers[BUDGET_EXCEEDED_HEADER]


async def test_budget_breach_rejects_with_429_and_the_documented_header() -> None:
    router = Router(build_config("reject"))
    app = build_app(router)
    async with client(app) as http:
        for _ in range(2):
            allowed = await http.post(
                "/v1/chat/completions", json=body(), headers={"x-api-key": "k2"}
            )
            assert allowed.status_code == 200
        rejected = await http.post("/v1/chat/completions", json=body(), headers={"x-api-key": "k2"})

    assert rejected.status_code == 429
    header = rejected.headers[BUDGET_EXCEEDED_HEADER]
    assert "action=reject" in header
    assert "limit=1.00" in header
    assert rejected.json()["error"] == "budget_exceeded"


async def test_one_key_under_concurrent_load_accounts_exactly() -> None:
    router = Router(build_config("downgrade_tier"))
    app = build_app(router)
    async with client(app) as http:
        responses = await asyncio.gather(
            *(
                http.post("/v1/chat/completions", json=body(), headers={"x-api-key": "k3"})
                for _ in range(25)
            )
        )

    assert all(response.status_code == 200 for response in responses)
    assert await router.budget.ledger.spent("k3") == pytest.approx(25 * COST_PER_CALL)
    # Every request charged exactly once, so the running totals are the
    # multiples of the per-call cost with nothing lost to a lost update.
    totals = sorted(response.json()["spent"] for response in responses)
    assert totals == pytest.approx([COST_PER_CALL * n for n in range(1, 26)])


async def test_separate_keys_have_separate_budgets_under_load() -> None:
    router = Router(build_config("reject"))
    app = build_app(router)
    async with client(app) as http:
        # k4 burns its budget; k5 must be untouched by that.
        for _ in range(2):
            await http.post("/v1/chat/completions", json=body(), headers={"x-api-key": "k4"})
        blocked, allowed = await asyncio.gather(
            http.post("/v1/chat/completions", json=body(), headers={"x-api-key": "k4"}),
            http.post("/v1/chat/completions", json=body(), headers={"x-api-key": "k5"}),
        )

    assert blocked.status_code == 429
    assert allowed.status_code == 200
    assert allowed.json()["tier"] == "complex"


async def test_route_uses_the_tie_break_provider_and_caches_across_requests() -> None:
    """The router hands the tie-break the cheapest trivial model, once per prompt."""
    provider = FakeProvider("complex")
    router = Router(build_config("downgrade_tier"), tiebreak_provider=provider)
    ambiguous = CompletionRequest.model_validate(
        body(
            "Northwind Hospitals Group, renewal Q4 2026, policy section 3.2, incumbent "
            "contract, procurement freeze until October, champion on leave, no quote yet, "
            "deal desk notified."
        )
    )
    first = await router.route(ambiguous, api_key="k6")
    second = await router.route(ambiguous, api_key="k6")

    assert provider.call_count == 1
    assert provider.calls[0][1] == "cheap"
    assert first.tier is Complexity.COMPLEX
    assert second.classification.source == "cache"


async def test_pinned_model_leads_the_chain_but_keeps_the_tier_fallbacks() -> None:
    router = Router(build_config("downgrade_tier"), ledger=SpendLedger())
    req = CompletionRequest.model_validate({**body(), "model": "cheap"})
    route = await router.route(req, api_key="k7")
    assert route.primary == "cheap"
    assert route.chain == ["cheap", "dear"]
    assert route.tier is Complexity.COMPLEX
