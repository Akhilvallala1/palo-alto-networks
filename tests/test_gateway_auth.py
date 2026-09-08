"""Unit tests for API-key resolution.

Auth is the first thing that touches a request and the only thing standing
between the open internet and a spend ledger, so its edges are tested directly
rather than only through the HTTP surface.
"""

import pytest

from conduit.contracts import Complexity
from conduit.gateway.auth import ANONYMOUS, ApiKeyDirectory, extract_key
from conduit.gateway.errors import UnauthorizedError
from conduit.gateway.settings import ApiKeySettings, GatewaySettings
from tests.gateway_fixtures import GOOD_KEY, SLOW_KEY, TRIVIAL_KEY, gateway_settings


def test_resolves_a_known_key_to_its_team_and_quota() -> None:
    directory = ApiKeyDirectory(gateway_settings())
    principal = directory.resolve(GOOD_KEY)
    assert principal.team == "acme"
    assert principal.max_tier is Complexity.COMPLEX


def test_a_key_without_its_own_rate_limit_inherits_the_default() -> None:
    directory = ApiKeyDirectory(gateway_settings(rpm=17))
    assert directory.resolve(GOOD_KEY).rate_limit_per_minute == 17
    # ...and one that sets its own keeps it.
    assert directory.resolve(SLOW_KEY).rate_limit_per_minute == 2


def test_an_unknown_key_is_rejected() -> None:
    directory = ApiKeyDirectory(gateway_settings())
    with pytest.raises(UnauthorizedError):
        directory.resolve("not-a-key")


def test_a_missing_key_is_rejected_and_names_both_headers() -> None:
    directory = ApiKeyDirectory(gateway_settings())
    with pytest.raises(UnauthorizedError) as exc:
        directory.resolve(None)
    assert "X-API-Key" in exc.value.message
    assert "Bearer" in exc.value.message


def test_require_auth_false_yields_the_unmetered_anonymous_principal() -> None:
    directory = ApiKeyDirectory(gateway_settings(require_auth=False))
    principal = directory.resolve(None)
    assert principal == ANONYMOUS
    # An empty key is what tells the router not to meter this caller at all.
    assert principal.api_key == ""


def test_a_gateway_with_no_keys_refuses_to_start() -> None:
    with pytest.raises(ValueError, match="require_auth"):
        ApiKeyDirectory(GatewaySettings(api_keys=[], require_auth=True))


def test_duplicate_keys_are_a_config_error() -> None:
    with pytest.raises(ValueError, match="duplicate api key"):
        GatewaySettings(
            api_keys=[
                ApiKeySettings(key="same", team="a"),
                ApiKeySettings(key="same", team="b"),
            ]
        )


def test_budget_limits_only_lists_keys_that_set_one() -> None:
    directory = ApiKeyDirectory(gateway_settings())
    assert directory.budget_limits() == {TRIVIAL_KEY: 0.0}


def test_fingerprint_is_stable_and_never_the_key() -> None:
    directory = ApiKeyDirectory(gateway_settings())
    principal = directory.resolve(GOOD_KEY)
    assert principal.fingerprint == directory.resolve(GOOD_KEY).fingerprint
    assert GOOD_KEY not in principal.fingerprint


@pytest.mark.parametrize(
    ("x_api_key", "authorization", "expected"),
    [
        ("k", None, "k"),
        (None, "Bearer k", "k"),
        (None, "bearer k", "k"),  # the OpenAI SDK's casing is not guaranteed
        ("k", "Bearer other", "k"),  # X-API-Key wins
        (None, "Basic k", None),  # not a scheme we accept
        (None, "Bearer   ", None),  # empty after the scheme
        ("  ", None, None),
        (None, None, None),
    ],
)
def test_extract_key_reads_both_header_styles(
    x_api_key: str | None, authorization: str | None, expected: str | None
) -> None:
    assert extract_key(x_api_key, authorization) == expected
