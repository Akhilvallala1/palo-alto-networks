"""Registry tests: the model->provider join, dormancy, tier filtering, pricing.

The registry is the only place that knows a price, so these assert against
`config/models.yaml` rather than against numbers written in the test.
"""

import pytest

from conduit.contracts import Complexity
from conduit.providers.registry import (
    ALL_PROVIDERS,
    PROVIDER_KEY_ENV,
    ModelRegistry,
    UnknownModelError,
    available_providers,
    build_providers,
    build_registry,
)
from provider_fixtures import ECHO, FLASH, GPT, HAIKU, LLAMA, OPUS, REPO_CONFIG, SONNET

KEYLESS_ENV: dict[str, str] = {}
ALL_KEYS_ENV = {
    "ANTHROPIC_API_KEY": "a",
    "OPENAI_API_KEY": "o",
    "GOOGLE_API_KEY": "g",
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
    reg = registry(ALL_KEYS_ENV)
    assert set(reg.provider_names) == set(ALL_PROVIDERS)
    assert {HAIKU, SONNET, OPUS, LLAMA, ECHO, GPT, FLASH} == set(reg.model_ids)


def test_anthropic_is_dormant_without_its_key_so_zero_key_runs_land_on_local() -> None:
    reg = registry(KEYLESS_ENV)
    assert reg.provider_names == ("ollama", "mock")
    assert set(reg.model_ids) == {LLAMA, ECHO}


def test_keyless_providers_are_always_available() -> None:
    assert available_providers({}) == ("ollama", "mock")
    assert "ollama" in available_providers(ALL_KEYS_ENV)


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
