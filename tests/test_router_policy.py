"""Unit tests for tier -> model chain resolution and registry pricing."""

import pytest

from conduit.config import ModelsConfig, ModelSpec, RoutingConfig, load_config
from conduit.contracts import Complexity, Usage
from conduit.router.policy import PolicyError, RoutingPolicy, price_usd


def routing(**tiers: list[str]) -> RoutingConfig:
    return RoutingConfig.model_validate(
        {
            "tiers": {
                "trivial": tiers.get("trivial", ["cheap"]),
                "standard": tiers.get("standard", ["mid"]),
                "complex": tiers.get("complex", ["dear"]),
            },
            "budgets": {"default_daily_usd": 5.0, "on_exceed": "downgrade_tier"},
        }
    )


REGISTRY = ModelsConfig(
    models={
        "cheap": ModelSpec(
            provider="fake",
            input_per_1m_usd=1.0,
            output_per_1m_usd=5.0,
            context_window=1000,
            tiers=[Complexity.TRIVIAL, Complexity.STANDARD],
        ),
        "mid": ModelSpec(
            provider="fake",
            input_per_1m_usd=2.0,
            output_per_1m_usd=10.0,
            context_window=1000,
            tiers=[Complexity.STANDARD, Complexity.COMPLEX],
        ),
        "dear": ModelSpec(
            provider="fake",
            input_per_1m_usd=5.0,
            output_per_1m_usd=25.0,
            context_window=1000,
            tiers=[Complexity.COMPLEX],
        ),
        "free": ModelSpec(
            provider="fake",
            input_per_1m_usd=0.0,
            output_per_1m_usd=0.0,
            context_window=1000,
            tiers=[Complexity.TRIVIAL],
        ),
    }
)


def test_chain_preserves_config_order() -> None:
    policy = RoutingPolicy(routing(trivial=["cheap", "free"]))
    chain = policy.chain_for(Complexity.TRIVIAL)
    assert chain.models == ["cheap", "free"]
    assert chain.primary == "cheap"
    assert policy.primary_for(Complexity.TRIVIAL) == "cheap"


def test_chain_is_a_copy_so_callers_cannot_mutate_config() -> None:
    policy = RoutingPolicy(routing(trivial=["cheap", "free"]))
    policy.chain_for(Complexity.TRIVIAL).models.append("dear")
    assert policy.chain_for(Complexity.TRIVIAL).models == ["cheap", "free"]


def test_downgrade_walks_one_rung_and_floors_at_trivial() -> None:
    assert RoutingPolicy.downgrade(Complexity.COMPLEX) is Complexity.STANDARD
    assert RoutingPolicy.downgrade(Complexity.STANDARD) is Complexity.TRIVIAL
    assert RoutingPolicy.downgrade(Complexity.TRIVIAL) is Complexity.TRIVIAL


def test_upgrade_walks_one_rung_and_caps_at_complex() -> None:
    assert RoutingPolicy.upgrade(Complexity.TRIVIAL) is Complexity.STANDARD
    assert RoutingPolicy.upgrade(Complexity.COMPLEX) is Complexity.COMPLEX


def test_validation_rejects_an_unknown_model() -> None:
    with pytest.raises(PolicyError, match="unknown model"):
        RoutingPolicy(routing(trivial=["ghost"]), REGISTRY)


def test_validation_rejects_a_tier_ineligible_model() -> None:
    with pytest.raises(PolicyError, match="not eligible"):
        RoutingPolicy(routing(trivial=["dear"]), REGISTRY)


def test_cheapest_model_prefers_price_then_chain_order() -> None:
    policy = RoutingPolicy(routing(trivial=["cheap", "free"]), REGISTRY)
    assert policy.cheapest_model(Complexity.TRIVIAL) == "free"


def test_cheapest_model_without_a_registry_falls_back_to_the_primary() -> None:
    policy = RoutingPolicy(routing(trivial=["cheap", "free"]))
    assert policy.cheapest_model(Complexity.TRIVIAL) == "cheap"


def test_price_uses_registry_rates_and_cache_multipliers() -> None:
    spec = REGISTRY.models["dear"]  # $5/1M in, $25/1M out
    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        cost_usd=0.0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    # 5 + 25 + (5 * 0.1) + (5 * 1.25)
    assert price_usd(spec, usage) == pytest.approx(36.75)


def test_policy_price_rejects_an_unknown_model() -> None:
    policy = RoutingPolicy(routing(), REGISTRY)
    usage = Usage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0)
    with pytest.raises(PolicyError, match="unknown model"):
        policy.price("ghost", usage)


def test_policy_price_needs_a_registry() -> None:
    policy = RoutingPolicy(routing())
    usage = Usage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0)
    with pytest.raises(PolicyError, match="no model registry"):
        policy.price("cheap", usage)


def test_shipped_config_is_internally_consistent() -> None:
    """config/routing.yaml chains must all be eligible in config/models.yaml."""
    config = load_config()
    policy = RoutingPolicy(config.routing, config.models)
    for tier in Complexity:
        assert policy.chain_for(tier).models
