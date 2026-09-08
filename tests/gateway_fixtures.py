"""Builders for a fully-wired gateway that never touches the network.

Every test app is assembled from the same real components the production app
uses — a real `GuardChain`, a real `Router`, a real `FailoverChain`, a real
`Recorder` — with only the provider layer swapped for deterministic doubles.
That is deliberate: the point of #6's tests is that the four wave-2 packages
work *together*, so stubbing any of them out would test the stub.

Prices are non-zero and deliberately round, so a cost assertion reads as
arithmetic rather than as a magic number: 1.00 USD per 1M input tokens and 2.00
per 1M output.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from conduit.config import (
    BudgetSettings,
    ConduitConfig,
    GuardsConfig,
    ModelsConfig,
    ModelSpec,
    RoutingConfig,
)
from conduit.contracts import CompletionRequest, CompletionResponse, Complexity, Usage
from conduit.gateway.app import create_app
from conduit.gateway.settings import ApiKeySettings, GatewaySettings
from conduit.providers.base import ProviderUnavailableError
from conduit.providers.mock import MockProvider
from conduit.providers.registry import ModelRegistry
from conduit.telemetry.recorder import Recorder
from conduit.telemetry.settings import TelemetrySettings
from conduit.telemetry.store import JsonlSqliteStore

__all__ = [
    "ECHO",
    "GOOD_KEY",
    "SECONDARY",
    "SLOW_KEY",
    "TRIVIAL_KEY",
    "ScriptedProvider",
    "build_app",
    "gateway_settings",
    "make_config",
    "make_recorder",
    "make_registry",
]

ECHO = "mock:echo"
SECONDARY = "scripted:backup"

#: Keys the fixture apps are built with. Distinct quotas so one app can exercise
#: auth, rate limiting and the tier ceiling without three configurations.
GOOD_KEY = "test-key"
SLOW_KEY = "slow-key"
TRIVIAL_KEY = "trivial-key"


def make_config(
    *,
    on_exceed: str = "downgrade_tier",
    daily_usd: float = 5.0,
    guards: GuardsConfig | None = None,
) -> ConduitConfig:
    """Routing and pricing over two models, both offline.

    Every tier resolves to `mock:echo` first so a request lands on a real
    adapter, with `scripted:backup` behind it to give failover somewhere to go.
    """
    tiers = [Complexity.TRIVIAL, Complexity.STANDARD, Complexity.COMPLEX]
    models = ModelsConfig(
        models={
            ECHO: ModelSpec(
                provider="mock",
                input_per_1m_usd=1.0,
                output_per_1m_usd=2.0,
                context_window=200_000,
                tiers=tiers,
            ),
            SECONDARY: ModelSpec(
                provider="scripted",
                input_per_1m_usd=1.0,
                output_per_1m_usd=2.0,
                context_window=200_000,
                tiers=tiers,
            ),
        }
    )
    return ConduitConfig(
        routing=RoutingConfig(
            tiers={tier: [ECHO, SECONDARY] for tier in tiers},
            budgets=BudgetSettings(default_daily_usd=daily_usd, on_exceed=on_exceed),
        ),
        models=models,
        guards=guards if guards is not None else GuardsConfig(),
    )


class ScriptedProvider:
    """A `Provider` whose every behaviour is dictated by the test.

    Implements the protocol by hand rather than subclassing `BaseProvider`, so a
    test can make it violate provider invariants — return a wrong `routed_tier`,
    fail on the first call and succeed on the second — which is exactly what the
    failover and tier-attribution assertions need.
    """

    name = "scripted"

    def __init__(
        self,
        *,
        text: str = "scripted reply",
        fail_times: int = 0,
        healthy: bool = True,
        latency_ms: int = 7,
        models: Sequence[str] = (SECONDARY,),
    ) -> None:
        self.text = text
        self.fail_times = fail_times
        self.healthy = healthy
        self.latency_ms = latency_ms
        self._models = tuple(models)
        self.calls: list[tuple[CompletionRequest, str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def supports(self, model: str) -> bool:
        return model in self._models

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        self.calls.append((req, model))
        if len(self.calls) <= self.fail_times:
            raise ProviderUnavailableError(self.name, "scripted failure")
        return CompletionResponse(
            text=self.text,
            model=model,
            provider=self.name,
            usage=Usage(prompt_tokens=100, completion_tokens=50, cost_usd=999.0),
            # The provider's own measurement of its own call, which the recorder
            # must keep in `provider_latency_ms` rather than report as latency.
            latency_ms=self.latency_ms,
            # Deliberately wrong: a bare adapter cannot know the routed tier, so
            # this is what `FailoverChain` must overwrite.
            routed_tier=Complexity.TRIVIAL,
        )

    async def health(self) -> bool:
        return self.healthy


def make_registry(
    config: ConduitConfig,
    *,
    mock_healthy: bool = True,
    scripted: ScriptedProvider | None = None,
) -> ModelRegistry:
    """The two offline adapters, joined to the config's price table."""
    served = {mid: spec for mid, spec in config.models.models.items() if spec.provider == "mock"}
    mock: Any = MockProvider(models=served)
    if not mock_healthy:
        mock = _UnreachableMock(models=served)
    return ModelRegistry(
        config.models.models,
        [mock, scripted if scripted is not None else ScriptedProvider()],
    )


class _UnreachableMock(MockProvider):
    """A mock adapter that is up but reports itself unreachable, for /readyz."""

    async def _probe(self) -> bool:
        return False


def make_recorder(tmp_path: Path, config: ConduitConfig) -> Recorder:
    """A real recorder over a real store, in a temp directory."""
    store = JsonlSqliteStore(tmp_path / "events.jsonl", tmp_path / "events.db")
    return Recorder(
        store,
        models=config.models.models,
        settings=TelemetrySettings(enabled=True, dir=tmp_path, persist_text=False),
    )


def gateway_settings(*, require_auth: bool = True, rpm: int = 60) -> GatewaySettings:
    """Three keys: a normal one, a nearly-rate-limited one, a tier-capped one."""
    return GatewaySettings(
        api_keys=[
            ApiKeySettings(key=GOOD_KEY, team="acme", max_tier=Complexity.COMPLEX),
            ApiKeySettings(key=SLOW_KEY, team="batch", rate_limit_per_minute=2),
            ApiKeySettings(
                key=TRIVIAL_KEY,
                team="cheap",
                max_tier=Complexity.TRIVIAL,
                daily_budget_usd=0.0,
            ),
        ],
        rate_limit_per_minute=rpm,
        require_auth=require_auth,
    )


def build_app(
    tmp_path: Path,
    *,
    config: ConduitConfig | None = None,
    settings: GatewaySettings | None = None,
    registry: ModelRegistry | None = None,
    recorder: Recorder | None = None,
    **kwargs: Any,
) -> FastAPI:
    """A gateway wired end to end, offline, with everything else real."""
    config = config if config is not None else make_config()
    return create_app(
        config=config,
        gateway=settings if settings is not None else gateway_settings(),
        registry=registry if registry is not None else make_registry(config),
        recorder=recorder if recorder is not None else make_recorder(tmp_path, config),
        setup_logging=False,
        **kwargs,
    )
