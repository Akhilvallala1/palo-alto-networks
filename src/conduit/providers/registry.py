"""The model registry: model id -> provider, price, context window, tiers.

Every fact about a model comes from `config/models.yaml`. This module adds no
prices of its own; it joins that table to live provider instances and answers
the questions the router and the failover chain ask.

**Dormancy.** A provider is only instantiated when its credential is present in
the environment, and any model whose provider is absent is dropped from the
registry entirely. So an unset `OPENAI_API_KEY` does not merely disable the
OpenAI adapter, it makes every OpenAI model invisible to routing — which is the
behaviour that lets the whole gateway run with zero keys on `ollama` + `mock`.
"""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import httpx

from conduit.config import ModelsConfig, ModelSpec, load_models
from conduit.contracts import Complexity, Provider

from . import anthropic as anthropic_adapter
from . import gemini as gemini_adapter
from . import openai as openai_adapter
from .base import BaseProvider, HttpProvider, RetryPolicy, price_usage
from .mock import MockProvider
from .ollama import OllamaProvider

__all__ = [
    "ALL_PROVIDERS",
    "KEYLESS_PROVIDERS",
    "PROVIDER_KEY_ENV",
    "ModelRegistry",
    "UnknownModelError",
    "available_providers",
    "build_providers",
    "build_registry",
]

#: Every adapter this package ships, in the order they are considered.
ALL_PROVIDERS: tuple[str, ...] = ("anthropic", "ollama", "mock", "openai", "gemini")

#: Backends that need no credential. These are why the repo is runnable with
#: zero keys, which matters more for a reviewer than any hosted vendor.
KEYLESS_PROVIDERS = frozenset({"ollama", "mock"})

PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": anthropic_adapter.API_KEY_ENV,
    "openai": openai_adapter.API_KEY_ENV,
    "gemini": gemini_adapter.API_KEY_ENV,
}

_HTTP_ADAPTERS: dict[str, Callable[..., HttpProvider]] = {
    "anthropic": anthropic_adapter.AnthropicProvider,
    "ollama": OllamaProvider,
    "openai": openai_adapter.OpenAIProvider,
    "gemini": gemini_adapter.GeminiProvider,
}


class UnknownModelError(LookupError):
    """A model id that is not in the registry — unpriced, or provider dormant."""


class ModelRegistry:
    """The joined view of `config/models.yaml` and the live provider set."""

    def __init__(self, models: Mapping[str, ModelSpec], providers: Iterable[Provider]) -> None:
        self._providers: dict[str, Provider] = {p.name: p for p in providers}
        self._models: dict[str, ModelSpec] = {
            model_id: spec for model_id, spec in models.items() if spec.provider in self._providers
        }

    # -- Membership --------------------------------------------------------- #

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._models

    def __len__(self) -> int:
        return len(self._models)

    @property
    def model_ids(self) -> tuple[str, ...]:
        """Registered models, in `config/models.yaml` order."""
        return tuple(self._models)

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(self._providers)

    @property
    def providers(self) -> tuple[Provider, ...]:
        return tuple(self._providers.values())

    # -- Lookups ------------------------------------------------------------ #

    def spec(self, model_id: str) -> ModelSpec:
        spec = self._models.get(model_id)
        if spec is None:
            raise UnknownModelError(
                f"{model_id!r} is not registered; known models: {', '.join(self._models) or 'none'}"
            )
        return spec

    def provider_for(self, model_id: str) -> Provider:
        """The adapter that serves `model_id`, or `UnknownModelError`."""
        return self._providers[self.spec(model_id).provider]

    def provider(self, name: str) -> Provider:
        provider = self._providers.get(name)
        if provider is None:
            raise UnknownModelError(f"provider {name!r} is not registered")
        return provider

    def context_window(self, model_id: str) -> int:
        return self.spec(model_id).context_window

    # -- Tier eligibility --------------------------------------------------- #

    def is_eligible(self, model_id: str, tier: Complexity) -> bool:
        return tier in self.spec(model_id).tiers

    def models_for_tier(self, tier: Complexity) -> tuple[str, ...]:
        """Registered models this tier may route to, in config order."""
        return tuple(mid for mid, spec in self._models.items() if tier in spec.tiers)

    def usable_chain(self, chain: Sequence[str], tier: Complexity | None = None) -> tuple[str, ...]:
        """Drop chain entries that are unregistered or ineligible for the tier.

        A routing chain names models by hand, so it will happily name one whose
        provider is dormant. Filtering here keeps that from becoming a runtime
        `UnknownModelError` on the hot path.
        """
        return tuple(
            model_id
            for model_id in chain
            if model_id in self._models and (tier is None or self.is_eligible(model_id, tier))
        )

    # -- Pricing ------------------------------------------------------------ #

    def price(
        self,
        model_id: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Cost in USD, from the YAML prices alone."""
        return price_usage(
            self.spec(model_id),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def available_providers(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Provider names whose credential is present, plus the keyless ones."""
    source = os.environ if env is None else env
    return tuple(
        name
        for name in ALL_PROVIDERS
        if name in KEYLESS_PROVIDERS or source.get(PROVIDER_KEY_ENV[name])
    )


def build_providers(
    models: Mapping[str, ModelSpec],
    *,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = 30.0,
    retry: RetryPolicy | None = None,
) -> tuple[BaseProvider, ...]:
    """Instantiate every available adapter that has at least one model to serve.

    Each adapter is handed only the rows of the registry it owns, so
    `supports()` is config-driven and no adapter can claim a model that pricing
    does not know about.
    """
    built: list[BaseProvider] = []
    for name in available_providers(env):
        served = {mid: spec for mid, spec in models.items() if spec.provider == name}
        if not served:
            continue
        if name == "mock":
            built.append(MockProvider(models=served, timeout_s=timeout_s, retry=retry))
        else:
            built.append(
                _HTTP_ADAPTERS[name](
                    models=served,
                    env=env,
                    transport=transport,
                    timeout_s=timeout_s,
                    retry=retry,
                )
            )
    return tuple(built)


def build_registry(
    *,
    models: ModelsConfig | None = None,
    config_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = 30.0,
    retry: RetryPolicy | None = None,
) -> ModelRegistry:
    """Load `models.yaml` and wire it to the adapters the environment allows."""
    table = (models if models is not None else load_models(config_dir, env)).models
    return ModelRegistry(
        table,
        build_providers(table, env=env, transport=transport, timeout_s=timeout_s, retry=retry),
    )
