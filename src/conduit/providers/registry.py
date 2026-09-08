"""The model registry: model id -> provider, price, context window, tiers.

Every fact about a model comes from `config/models.yaml`. This module adds no
prices of its own; it joins that table to live provider instances and answers
the questions the router and the failover chain ask.

**Dormancy.** A provider is only instantiated when its credential is present in
the environment, and any model whose provider is absent is dropped from the
registry entirely. So an unset `OPENAI_API_KEY` does not merely disable the
OpenAI adapter, it makes every OpenAI model invisible to routing — which is the
behaviour that lets the whole gateway run with zero keys on `ollama` + `mock`.

Dormancy is deliberate; a *typo* is not. `models.yaml` naming a provider no
adapter implements used to drop that model by the same code path, so a
misspelled `anthropc` produced a registry that was quietly one model short and
a chain that silently skipped a hop. `validate_provider_names` separates the
two: an absent credential is dormancy, an unimplementable name is an error.
"""

import logging
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
    "ALLOW_MOCK_ENV",
    "ALL_PROVIDERS",
    "FABRICATING_PROVIDERS",
    "KEYLESS_PROVIDERS",
    "PROVIDER_KEY_ENV",
    "ModelRegistry",
    "UnknownModelError",
    "UnknownProviderError",
    "available_providers",
    "build_providers",
    "build_registry",
    "keyed_providers",
    "validate_provider_names",
]

_log = logging.getLogger(__name__)

#: Every adapter this package ships, in the order they are considered.
ALL_PROVIDERS: tuple[str, ...] = ("anthropic", "ollama", "mock", "openai", "gemini")

#: Backends that need no credential. These are why the repo is runnable with
#: zero keys, which matters more for a reviewer than any hosted vendor.
KEYLESS_PROVIDERS = frozenset({"ollama", "mock"})

#: Keyless is not the same as safe. `ollama` performs real inference; `mock`
#: fabricates a completion from the prompt. Every chain in `config/routing.yaml`
#: terminates in `mock:echo` so that a zero-key checkout still serves (AC-17) —
#: which means that in a deployment holding a real credential, an upstream
#: outage would walk the chain down to `mock` and hand the caller invented text
#: at `cost_usd: 0.0` with a 200. A quote workflow cannot tell that apart from
#: an answer. So `mock` goes dormant as soon as any real credential exists, and
#: an exhausted chain fails loudly instead.
FABRICATING_PROVIDERS = frozenset({"mock"})

#: Escape hatch for the one legitimate case: a test or a demo that deliberately
#: exercises the mock path while a real key happens to be in the environment.
ALLOW_MOCK_ENV = "CONDUIT_ALLOW_MOCK"

#: Values that mean "off" for `ALLOW_MOCK_ENV`. An unset var is also off.
_FALSEY = frozenset({"", "0", "false", "no", "off"})

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


class UnknownProviderError(ValueError):
    """`models.yaml` names a provider no adapter implements. Almost always a typo."""


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


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in _FALSEY


def keyed_providers(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Providers holding a real credential — the test for "this is a deployment"."""
    source = os.environ if env is None else env
    return tuple(
        name
        for name in ALL_PROVIDERS
        if name in PROVIDER_KEY_ENV and source.get(PROVIDER_KEY_ENV[name])
    )


def available_providers(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Provider names whose credential is present, plus the keyless ones.

    `mock` is the exception to "keyless means always on". It is available only
    when no real credential exists anywhere in the environment, or when
    `CONDUIT_ALLOW_MOCK` is set explicitly. Fabricated text reaching a caller
    that believes it is talking to a model is worse than a 502, and the failover
    chain has no way to distinguish the two on its own.
    """
    source = os.environ if env is None else env
    keyed = keyed_providers(source)
    override = _enabled(source.get(ALLOW_MOCK_ENV))
    fabricating_ok = not keyed or override
    if keyed and override:
        _log.warning(
            "%s is set while %s holds a credential: mock:echo stays routable and may "
            "answer a real request with fabricated text.",
            ALLOW_MOCK_ENV,
            ", ".join(keyed),
        )
    return tuple(
        name
        for name in ALL_PROVIDERS
        if (name in KEYLESS_PROVIDERS and (fabricating_ok or name not in FABRICATING_PROVIDERS))
        or (name in PROVIDER_KEY_ENV and source.get(PROVIDER_KEY_ENV[name]))
    )


def validate_provider_names(models: Mapping[str, ModelSpec]) -> None:
    """Fail on a provider name no adapter implements (issue #11).

    Dormancy drops a model whose provider has no key, which is intended and
    silent. A misspelled provider took the identical path, so `anthropc` cost
    you a model and a chain hop with nothing logged anywhere. There is no
    deployment in which that name is correct, so it is a config error.
    """
    unknown = {
        model_id: spec.provider
        for model_id, spec in models.items()
        if spec.provider not in ALL_PROVIDERS
    }
    if unknown:
        detail = ", ".join(f"{model_id} -> {name!r}" for model_id, name in sorted(unknown.items()))
        raise UnknownProviderError(
            f"no adapter implements the provider named by {len(unknown)} model(s): {detail}. "
            f"Known providers: {', '.join(ALL_PROVIDERS)}."
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
    # On the config path only. `ModelRegistry` itself stays generic: tests and the
    # eval plane hand it providers named "scripted" or "fake", and those are real
    # objects being injected, not names that have to resolve to a shipped adapter.
    validate_provider_names(table)
    return ModelRegistry(
        table,
        build_providers(table, env=env, transport=transport, timeout_s=timeout_s, retry=retry),
    )
