"""Unit tests for error mapping (AC-8).

The rule under test is that a caller sees Conduit's own vocabulary and nothing
else: no tracebacks, no exception class names, no vendor response bodies. Two of
these tests exist specifically because upstream exception *messages* are unsafe
to forward — `AllProvidersFailedError` embeds the last provider error verbatim.
"""

from datetime import date
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conduit.contracts import Complexity, GuardVerdict
from conduit.gateway.errors import (
    INTERNAL_MESSAGE,
    BadRequestError,
    GatewayError,
    GuardRejectedError,
    RateLimitExceededError,
    StreamingUnsupportedError,
    UnauthorizedError,
    UpstreamUnavailableError,
    error_payload,
    install_error_handlers,
)
from conduit.providers.base import ProviderUnavailableError
from conduit.providers.failover import AllProvidersFailedError
from conduit.router.cost import BUDGET_EXCEEDED_HEADER, BudgetDecision, BudgetExceededError

#: A provider error of the kind an HTTP adapter really raises: it quotes the
#: vendor's response body, an internal URL and a customer's own prompt.
LEAKY = (
    "anthropic 429: {'error': {'message': 'rate limited on account acct_918273, "
    "see https://internal.vendor/console'}} while completing 'my SSN is 123-45-6789'"
)


def budget_decision(*, spent: float = 5.4, limit: float = 5.0) -> BudgetDecision:
    return BudgetDecision(
        tier=Complexity.STANDARD,
        requested_tier=Complexity.COMPLEX,
        spent_usd=spent,
        limit_usd=limit,
        exceeded=True,
        window=date(2026, 9, 7),
    )


@pytest.fixture
def client() -> TestClient:
    """An app whose only job is to raise, so the handlers are what is tested."""
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/raise/{kind}")
    async def boom(kind: str) -> dict[str, str]:
        raise {
            "unauthorized": UnauthorizedError(),
            "ratelimit": RateLimitExceededError(30, 60),
            "streaming": StreamingUnsupportedError(),
            "bad": BadRequestError("model 'nope' is not configured"),
            "budget": BudgetExceededError(budget_decision(), "some-key"),
            "exhausted": AllProvidersFailedError(
                ["mock:echo", "scripted:backup"],
                ProviderUnavailableError("anthropic", LEAKY),
            ),
            "boom": ZeroDivisionError("division by zero in conduit/gateway/pipeline.py:412"),
        }[kind]

    # `raise_server_exceptions=False` is what lets the catch-all handler run
    # instead of TestClient re-raising the exception into the test.
    return TestClient(app, raise_server_exceptions=False)


def body(client: TestClient, kind: str) -> dict[str, Any]:
    payload: dict[str, Any] = client.get(f"/raise/{kind}").json()
    return payload


# -- the envelope ----------------------------------------------------------- #


def test_the_envelope_is_the_shape_the_openai_sdk_parses() -> None:
    payload = error_payload("nope", "invalid_request_error", code="bad_thing")
    assert payload == {
        "error": {
            "message": "nope",
            "type": "invalid_request_error",
            "param": None,
            "code": "bad_thing",
        }
    }


def test_the_code_defaults_to_the_type() -> None:
    assert error_payload("nope", "api_error")["error"]["code"] == "api_error"


def test_conduit_detail_is_namespaced_not_spread_over_the_envelope() -> None:
    payload = error_payload("nope", "api_error", conduit={"attempted": ["a"]})
    assert payload["error"]["conduit"] == {"attempted": ["a"]}


# -- status and header mapping ---------------------------------------------- #


def test_an_unauthorized_error_is_a_401_that_asks_for_a_bearer(client: TestClient) -> None:
    res = client.get("/raise/unauthorized")
    assert res.status_code == 401
    assert res.headers["WWW-Authenticate"] == "Bearer"
    assert res.json()["error"]["code"] == "invalid_api_key"


def test_a_rate_limit_error_is_a_429_with_retry_after(client: TestClient) -> None:
    res = client.get("/raise/ratelimit")
    assert res.status_code == 429
    assert res.headers["Retry-After"] == "30"
    assert res.json()["error"]["type"] == "rate_limit_error"
    assert res.json()["error"]["conduit"]["limit_per_minute"] == 60


def test_streaming_is_a_400_that_names_the_limitation(client: TestClient) -> None:
    res = client.get("/raise/streaming")
    assert res.status_code == 400
    assert "Streaming is not supported" in res.json()["error"]["message"]


def test_a_bad_request_is_a_400_carrying_the_callers_own_mistake(client: TestClient) -> None:
    res = client.get("/raise/bad")
    assert res.status_code == 400
    assert res.json()["error"]["message"] == "model 'nope' is not configured"


def test_a_guard_rejection_is_a_400_naming_the_category_and_the_guard() -> None:
    error = GuardRejectedError(
        GuardVerdict(allowed=False, risk_score=0.91, categories=["pii:us_ssn"]),
        "pii",
    )
    assert error.status_code == 400
    payload = error.payload()["error"]
    assert payload["code"] == "guard_rejected"
    assert "pii:us_ssn" in payload["message"]
    assert payload["conduit"] == {
        "guard": "pii",
        "stage": "ingress",
        "categories": ["pii:us_ssn"],
        "risk_score": 0.91,
    }


def test_a_guard_rejection_never_echoes_the_text_that_caused_it() -> None:
    """The offending prompt is the last thing that should ride back out."""
    verdict = GuardVerdict(
        allowed=False,
        risk_score=1.0,
        categories=["pii:us_ssn"],
        redacted_text="my SSN is <US_SSN_1>",
        entity_map={"<US_SSN_1>": "123-45-6789"},
    )
    rendered = str(GuardRejectedError(verdict, "pii").payload())
    assert "123-45-6789" not in rendered
    assert "<US_SSN_1>" not in rendered


# -- budget: #4's shape, rendered not redefined ----------------------------- #


def test_a_budget_breach_uses_the_routers_documented_header(client: TestClient) -> None:
    res = client.get("/raise/budget")
    assert res.status_code == 429
    assert res.headers[BUDGET_EXCEEDED_HEADER] == (
        "limit=5.00;spent=5.40;window=2026-09-07;action=reject"
    )


def test_a_budget_breach_explains_itself_in_the_json_too(client: TestClient) -> None:
    payload = body(client, "budget")["error"]
    assert payload["code"] == "budget_exceeded"
    assert "$5.00" in payload["message"]
    assert payload["conduit"]["action"] == "reject"
    assert payload["conduit"]["window"] == "2026-09-07"


# -- nothing internal escapes (AC-8) ---------------------------------------- #


def test_an_exhausted_chain_is_a_502_listing_only_our_own_model_ids(client: TestClient) -> None:
    res = client.get("/raise/exhausted")
    assert res.status_code == 502
    payload = res.json()["error"]
    assert payload["conduit"]["attempted"] == ["mock:echo", "scripted:backup"]


@pytest.mark.parametrize(
    "fragment", ["acct_918273", "internal.vendor", "123-45-6789", "rate limited"]
)
def test_the_provider_error_body_does_not_reach_the_caller(
    client: TestClient, fragment: str
) -> None:
    assert fragment not in client.get("/raise/exhausted").text


def test_an_unexpected_exception_is_a_500_with_no_traceback(client: TestClient) -> None:
    res = client.get("/raise/boom")
    assert res.status_code == 500
    assert res.json() == {
        "error": {
            "message": INTERNAL_MESSAGE,
            "type": "api_error",
            "param": None,
            "code": "internal_error",
        }
    }


@pytest.mark.parametrize("fragment", ["ZeroDivisionError", "Traceback", "pipeline.py", "division"])
def test_a_500_leaks_neither_the_exception_class_nor_the_file(
    client: TestClient, fragment: str
) -> None:
    assert fragment not in client.get("/raise/boom").text


def test_the_base_class_renders_a_response_without_an_app() -> None:
    """`GatewayError.response()` is used directly by the handlers; keep it honest."""
    error = GatewayError("something", code="c", headers={"X-Test": "1"})
    response = error.response()
    assert response.status_code == 500
    assert response.headers["X-Test"] == "1"


def test_an_upstream_error_reports_the_chain_it_tried() -> None:
    error = UpstreamUnavailableError(["a", "b"])
    assert error.status_code == 502
    assert error.payload()["error"]["conduit"]["attempted"] == ["a", "b"]
