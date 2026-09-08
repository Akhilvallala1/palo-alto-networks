"""Registry tests: the model->provider join, dormancy, tier filtering, pricing.

The registry is the only place that knows a price, so these assert against
`config/models.yaml` rather than against numbers written in the test.
"""

import logging

import pytest

from conduit.config import ModelSpec
from conduit.contracts import Complexity
from conduit.providers.registry import (
    ALL_PROVIDERS,
    ALLOW_MOCK_ENV,
    FABRICATING_PROVIDERS,
    PROVIDER_KEY_ENV,
    ModelRegistry,
    UnknownModelError,
    UnknownProviderError,
    available_providers,
    build_providers,
    build_registry,
    validate_provider_names,
)
from provider_fixtures import (
    ECHO,
    FLASH,
    GPT,
    HAIKU,
    LLAMA,
    OPUS,
    REPO_CONFIG,
    SONNET,
    repo_models,
)

KEYLESS_ENV: dict[str, str] = {}
#: Every adapter live at once, which the join, tier and pricing tests below need.
#: Since issue #14 that takes an explicit opt-in: a credential in the environment
#: makes `mock` dormant, so "all providers" and "all keys" are no longer the same
#: environment. The tests that care about the policy itself build their own env.
ALL_KEYS_ENV = {
    "ANTHROPIC_API_KEY": "a",
    "OPENAI_API_KEY": "o",
    "GOOGLE_API_KEY": "g",
    ALLOW_MOCK_ENV: "1",
}


def registry(env: dict[str, str]) -> ModelRegistry:
    return build_registry(config_dir=REPO_CONFIG, env=env)


# --------------------------------------------------------------------------- #
# AC-7: dormancy
# --------------------------------------------------------------------------- #


def test_dormant_providers_are_absent_when_their_key_is_unset() -> None:
    """AC-7. Not merely disabled — their models vanish from the registry."""
    reg = registry(KEYLESS_ENV)
    assert "openai" not in reg.provider_names
    assert "gemini" not in reg.provider_names
    assert GPT not in reg
    assert FLASH not in reg


def test_a_dormant_provider_wakes_up_when_its_key_appears() -> None:
    reg = registry({"OPENAI_API_KEY": "sk-test"})
    assert "openai" in reg.provider_names
    assert GPT in reg
    assert FLASH not in reg  # GOOGLE_API_KEY still unset


def test_every_provider_is_registered_when_every_key_is_present() -> None:
    """`ALL_KEYS_ENV` also opts mock back in; without that this is a 6-model registry."""
    reg = registry(ALL_KEYS_ENV)
    assert set(reg.provider_names) == set(ALL_PROVIDERS)
    assert {HAIKU, SONNET, OPUS, LLAMA, ECHO, GPT, FLASH} == set(reg.model_ids)


def test_anthropic_is_dormant_without_its_key_so_zero_key_runs_land_on_local() -> None:
    reg = registry(KEYLESS_ENV)
    assert reg.provider_names == ("ollama", "mock")
    assert set(reg.model_ids) == {LLAMA, ECHO}


def test_keyless_providers_carry_a_zero_key_checkout() -> None:
    """Both keyless backends serve when nothing is configured.

    Only `ollama` stays available once a credential appears; `mock` is held to a
    stricter rule by the issue #14 tests further down, because it invents text
    rather than inferring it.
    """
    assert available_providers({}) == ("ollama", "mock")
    assert "ollama" in available_providers({"ANTHROPIC_API_KEY": "sk-live"})


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_each_keyed_provider_names_the_env_var_that_wakes_it(provider: str) -> None:
    key = PROVIDER_KEY_ENV[provider]
    assert provider not in available_providers({})
    assert provider in available_providers({key: "value"})


def test_an_empty_key_does_not_count_as_present() -> None:
    assert "anthropic" not in available_providers({"ANTHROPIC_API_KEY": ""})


def test_a_provider_with_no_models_is_not_instantiated() -> None:
    models = {LLAMA: registry(KEYLESS_ENV).spec(LLAMA)}
    built = build_providers(models, env=ALL_KEYS_ENV)
    assert [p.name for p in built] == ["ollama"]


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #


def test_provider_for_returns_the_adapter_that_serves_the_model() -> None:
    reg = registry(ALL_KEYS_ENV)
    assert reg.provider_for(HAIKU).name == "anthropic"
    assert reg.provider_for(LLAMA).name == "ollama"
    assert reg.provider_for(ECHO).name == "mock"


def test_each_adapter_claims_exactly_its_own_models() -> None:
    reg = registry(ALL_KEYS_ENV)
    for model_id in reg.model_ids:
        provider = reg.provider_for(model_id)
        assert provider.supports(model_id)
        others = [m for m in reg.model_ids if reg.spec(m).provider != provider.name]
        assert not any(provider.supports(m) for m in others)


def test_an_unknown_model_raises_rather_than_defaulting() -> None:
    reg = registry(ALL_KEYS_ENV)
    with pytest.raises(UnknownModelError, match="not registered"):
        reg.spec("claude-nonexistent")
    with pytest.raises(UnknownModelError):
        reg.provider_for("claude-nonexistent")


def test_a_dormant_model_is_unknown_not_merely_unroutable() -> None:
    with pytest.raises(UnknownModelError):
        registry(KEYLESS_ENV).spec(GPT)


def test_an_unregistered_provider_name_raises() -> None:
    with pytest.raises(UnknownModelError, match="provider 'openai'"):
        registry(KEYLESS_ENV).provider("openai")


def test_context_windows_come_from_the_yaml() -> None:
    reg = registry(ALL_KEYS_ENV)
    assert reg.context_window(HAIKU) == 200_000
    assert reg.context_window(OPUS) == 1_000_000


def test_model_ids_preserve_config_order_and_length_agrees() -> None:
    reg = registry(ALL_KEYS_ENV)
    assert reg.model_ids[:3] == (OPUS, SONNET, HAIKU)
    assert len(reg) == len(reg.model_ids) == 7


# --------------------------------------------------------------------------- #
# Tier eligibility
# --------------------------------------------------------------------------- #


def test_models_for_tier_filters_on_eligibility() -> None:
    reg = registry(ALL_KEYS_ENV)
    assert reg.models_for_tier(Complexity.COMPLEX) == (OPUS, SONNET, ECHO, GPT)
    assert OPUS not in reg.models_for_tier(Complexity.TRIVIAL)


def test_tier_eligibility_is_per_model() -> None:
    reg = registry(ALL_KEYS_ENV)
    assert reg.is_eligible(SONNET, Complexity.COMPLEX)
    assert not reg.is_eligible(SONNET, Complexity.TRIVIAL)
    assert all(reg.is_eligible(ECHO, tier) for tier in Complexity)


def test_models_for_tier_hides_dormant_models() -> None:
    assert GPT not in registry(KEYLESS_ENV).models_for_tier(Complexity.COMPLEX)


def test_usable_chain_drops_models_the_registry_does_not_have() -> None:
    reg = registry(KEYLESS_ENV)
    assert reg.usable_chain([HAIKU, LLAMA, ECHO]) == (LLAMA, ECHO)


def test_usable_chain_can_also_enforce_tier_eligibility() -> None:
    reg = registry(ALL_KEYS_ENV)
    chain = [OPUS, SONNET, HAIKU]
    assert reg.usable_chain(chain) == (OPUS, SONNET, HAIKU)
    assert reg.usable_chain(chain, Complexity.COMPLEX) == (OPUS, SONNET)


def test_usable_chain_preserves_the_given_order() -> None:
    reg = registry(ALL_KEYS_ENV)
    assert reg.usable_chain([ECHO, LLAMA]) == (ECHO, LLAMA)
    assert reg.usable_chain([LLAMA, ECHO]) == (LLAMA, ECHO)


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


def test_registry_price_matches_the_yaml_rates() -> None:
    reg = registry(ALL_KEYS_ENV)
    spec = reg.spec(SONNET)
    cost = reg.price(SONNET, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(spec.input_per_1m_usd + spec.output_per_1m_usd)


def test_registry_price_accounts_for_cache_tokens() -> None:
    reg = registry(ALL_KEYS_ENV)
    cost = reg.price(
        HAIKU,
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost == pytest.approx(1.00 * 0.1 + 1.00 * 1.25)


def test_pricing_an_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError):
        registry(ALL_KEYS_ENV).price("nope", prompt_tokens=1, completion_tokens=1)


# --------------------------------------------------------------------------- #
# Issue #14: a fabricating provider must not be reachable in a keyed deployment
# --------------------------------------------------------------------------- #


def _respec(provider: str) -> ModelSpec:
    """An existing spec with its provider swapped, so field drift can't rot this."""
    return repo_models().models[OPUS].model_copy(update={"provider": provider})


def test_mock_is_available_when_no_credential_exists_anywhere() -> None:
    """AC-17: a zero-key checkout still has a terminal hop on every chain."""
    assert "mock" in available_providers(KEYLESS_ENV)


def test_mock_goes_dormant_as_soon_as_a_real_credential_is_present() -> None:
    """Issue #14. The failure this prevents is a 200, not a 500.

    Every chain in `config/routing.yaml` ends in `mock:echo`. Without this rule a
    keyed deployment whose vendor is down walks the chain to the end and answers
    with invented text at `cost_usd: 0.0`, which the caller cannot tell from a
    real completion. Dormant mock means the chain is exhausted and it fails loudly.
    """
    assert "mock" not in available_providers({"ANTHROPIC_API_KEY": "sk-live"})


@pytest.mark.parametrize("provider", sorted(PROVIDER_KEY_ENV))
def test_any_real_credential_is_enough_to_suppress_mock(provider: str) -> None:
    assert "mock" not in available_providers({PROVIDER_KEY_ENV[provider]: "sk-live"})


def test_ollama_stays_available_alongside_a_credential() -> None:
    """Keyless is not the property being restricted; fabricating is.

    Ollama performs real local inference, so a keyed deployment may still route
    to it. Suppressing it too would be the easy over-correction.
    """
    assert "ollama" in available_providers({"ANTHROPIC_API_KEY": "sk-live"})


def test_mock_can_be_forced_back_on_explicitly() -> None:
    assert "mock" in available_providers({"ANTHROPIC_API_KEY": "sk-live", ALLOW_MOCK_ENV: "1"})


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "FALSE", " off "])
def test_the_override_is_off_for_falsey_values(value: str) -> None:
    """An empty or "false" override must not read as "on"."""
    env = {"ANTHROPIC_API_KEY": "sk-live", ALLOW_MOCK_ENV: value}
    assert "mock" not in available_providers(env)


def test_forcing_mock_on_next_to_a_credential_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The dangerous configuration stays legal, but never silent."""
    with caplog.at_level(logging.WARNING, logger="conduit.providers.registry"):
        available_providers({"ANTHROPIC_API_KEY": "sk-live", ALLOW_MOCK_ENV: "1"})
    assert any("fabricated" in record.getMessage() for record in caplog.records)


def test_no_fabricating_model_survives_into_a_keyed_registry() -> None:
    """The end-to-end form of #14, asked the way the router asks it."""
    keyed = registry({"ANTHROPIC_API_KEY": "sk-live"})
    assert ECHO not in keyed
    for model_id in keyed.model_ids:
        assert keyed.spec(model_id).provider not in FABRICATING_PROVIDERS


def test_the_zero_key_registry_still_serves_echo() -> None:
    """The regression guard on the other side: #14 must not break AC-17."""
    assert ECHO in registry(KEYLESS_ENV)


# --------------------------------------------------------------------------- #
# Issue #11: a misspelled provider is a config error, not silent dormancy
# --------------------------------------------------------------------------- #


def test_a_misspelled_provider_raises_instead_of_dropping_the_model() -> None:
    """Issue #11. `anthropc` used to cost a model and a chain hop, silently."""
    with pytest.raises(UnknownProviderError) as excinfo:
        validate_provider_names({OPUS: _respec("anthropc")})
    message = str(excinfo.value)
    assert "anthropc" in message
    assert OPUS in message


def test_the_error_names_every_offending_model_not_just_the_first() -> None:
    models = {"a": _respec("anthropc"), "b": _respec("olama"), "c": _respec("mock")}
    with pytest.raises(UnknownProviderError) as excinfo:
        validate_provider_names(models)
    message = str(excinfo.value)
    assert "anthropc" in message
    assert "olama" in message
    assert "'mock'" not in message


def test_validation_accepts_every_provider_the_package_ships() -> None:
    validate_provider_names({name: _respec(name) for name in ALL_PROVIDERS})


def test_a_dormant_provider_is_not_a_validation_error() -> None:
    """The distinction the fix exists to draw.

    `openai` with no key is dormant and its models drop silently, which is
    intended. `openai` misspelled raises. Before this change both took the same
    path and produced the same registry, with opposite intent.
    """
    validate_provider_names({GPT: _respec("openai")})
    assert "openai" not in available_providers(KEYLESS_ENV)


def test_build_registry_rejects_a_typo_rather_than_returning_a_short_registry() -> None:
    broken = repo_models().model_copy(update={"models": {OPUS: _respec("anthropc")}})
    with pytest.raises(UnknownProviderError):
        build_registry(models=broken, env=ALL_KEYS_ENV)


def test_the_shipped_config_passes_its_own_validation() -> None:
    """Guards the guard: a typo in `config/models.yaml` now fails the suite."""
    validate_provider_names(repo_models().models)
