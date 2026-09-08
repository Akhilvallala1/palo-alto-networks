"""Tier -> ordered model chain, resolved from `config/routing.yaml`.

The chain is what the failover layer walks: index 0 is the primary, the rest
are fallbacks in order. This module is the only place that turns a `Complexity`
into model ids, and it owns the tier ladder (`downgrade`/`upgrade`) that the
budget engine uses.

Pricing lives in `config/models.yaml`, never in logic — `price_usd` reads it.
"""

from pydantic import BaseModel

from conduit.config import ModelsConfig, ModelSpec, RoutingConfig
from conduit.contracts import Complexity, Usage

__all__ = ["TIER_LADDER", "PolicyError", "RoutingPolicy", "TierChain", "price_usd"]

# Cheapest to most expensive. `downgrade` walks left and floors at trivial.
TIER_LADDER: tuple[Complexity, ...] = (
    Complexity.TRIVIAL,
    Complexity.STANDARD,
    Complexity.COMPLEX,
)

# Anthropic prompt-cache multipliers, per the `Usage` docstring in contracts.py.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


class PolicyError(Exception):
    """Raised when routing config and the model registry disagree."""


def price_usd(spec: ModelSpec, usage: Usage) -> float:
    """Price a `Usage` against a registry row.

    `prompt_tokens` is the uncached remainder, so cache reads and writes are
    priced separately off the input rate.
    """
    input_rate = spec.input_per_1m_usd / 1_000_000
    output_rate = spec.output_per_1m_usd / 1_000_000
    return (
        usage.prompt_tokens * input_rate
        + usage.completion_tokens * output_rate
        + usage.cache_read_tokens * input_rate * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
    )


class TierChain(BaseModel):
    """A resolved tier and the model chain to try, in order."""

    tier: Complexity
    models: list[str]

    @property
    def primary(self) -> str:
        return self.models[0]


class RoutingPolicy:
    """Resolves complexity tiers to model chains, and moves between tiers."""

    def __init__(self, routing: RoutingConfig, models: ModelsConfig | None = None) -> None:
        self.routing = routing
        self.models = models
        if models is not None:
            self.validate(models)

    def validate(self, models: ModelsConfig) -> None:
        """Fail loudly when a chain names an unknown or tier-ineligible model."""
        for tier, chain in self.routing.tiers.items():
            for model in chain:
                spec = models.models.get(model)
                if spec is None:
                    raise PolicyError(f"routing tier {tier.value!r} names unknown model {model!r}")
                if tier not in spec.tiers:
                    raise PolicyError(
                        f"model {model!r} is not eligible for tier {tier.value!r} "
                        f"(eligible: {', '.join(t.value for t in spec.tiers)})"
                    )

    def chain_for(self, tier: Complexity) -> TierChain:
        chain = self.routing.tiers.get(tier)
        if not chain:
            raise PolicyError(f"no model chain configured for tier {tier.value!r}")
        return TierChain(tier=tier, models=list(chain))

    def primary_for(self, tier: Complexity) -> str:
        return self.chain_for(tier).primary

    @staticmethod
    def downgrade(tier: Complexity) -> Complexity:
        """One rung cheaper, floored at trivial."""
        index = TIER_LADDER.index(tier)
        return TIER_LADDER[max(0, index - 1)]

    @staticmethod
    def upgrade(tier: Complexity) -> Complexity:
        """One rung dearer, capped at complex."""
        index = TIER_LADDER.index(tier)
        return TIER_LADDER[min(len(TIER_LADDER) - 1, index + 1)]

    def cheapest_model(self, tier: Complexity = Complexity.TRIVIAL) -> str:
        """The cheapest chain entry for a tier — what the tie-break runs on.

        Without a model registry, prices are unknown and the chain's primary is
        the documented preference, so it wins by default.
        """
        chain = self.chain_for(tier).models
        registry = self.models
        if registry is None:
            return chain[0]
        known = [model for model in chain if model in registry.models]
        if not known:
            return chain[0]

        def blended(model: str) -> float:
            spec = registry.models[model]
            return spec.input_per_1m_usd + spec.output_per_1m_usd

        return min(known, key=lambda model: (blended(model), chain.index(model)))

    def price(self, model: str, usage: Usage) -> float:
        """Cost of a usage record on a model, from the registry."""
        if self.models is None:
            raise PolicyError("no model registry loaded; cannot price a request")
        spec = self.models.models.get(model)
        if spec is None:
            raise PolicyError(f"unknown model {model!r} in registry")
        return price_usd(spec, usage)
