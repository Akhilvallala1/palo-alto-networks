"""The router's single entry point: request in, tier + model chain out.

`Router` is what the gateway (#6) holds. It wires the three pieces this issue
owns — classifier, policy, cost engine — and nothing else: the actual model
call belongs to the provider layer, which the router only knows through the
`Provider` protocol.

Usage from the gateway::

    route = await router.route(req, api_key=key)
    response = await failover.complete(req, route.chain)   # provider layer
    await router.record_spend(key, response.usage.cost_usd)
"""

from pathlib import Path

from pydantic import BaseModel

from conduit.config import ConduitConfig, load_config
from conduit.contracts import CompletionRequest, Complexity, Provider

from .classifier import Classification, Classifier
from .cost import BudgetDecision, BudgetEnforcer, SpendLedger
from .policy import RoutingPolicy
from .settings import ClassifierSettings, load_classifier_settings

__all__ = ["RouteResult", "Router"]


class RouteResult(BaseModel):
    """Everything the gateway needs to place the call and log the decision."""

    tier: Complexity
    chain: list[str]
    classification: Classification
    budget: BudgetDecision

    @property
    def primary(self) -> str:
        return self.chain[0]

    def headers(self) -> dict[str, str]:
        return self.budget.headers()


class Router:
    """Complexity classification, tier policy and budget enforcement."""

    def __init__(
        self,
        config: ConduitConfig,
        *,
        settings: ClassifierSettings | None = None,
        tiebreak_provider: Provider | None = None,
        ledger: SpendLedger | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or ClassifierSettings()
        self.policy = RoutingPolicy(config.routing, config.models)
        tiebreak_model = self.settings.tiebreak_model or self.policy.cheapest_model(
            Complexity.TRIVIAL
        )
        self.classifier = Classifier(
            self.settings,
            provider=tiebreak_provider,
            tiebreak_model=tiebreak_model,
        )
        self.budget = BudgetEnforcer(config.routing.budgets, self.policy, ledger)

    @classmethod
    def from_config_dir(
        cls,
        config_dir: Path | None = None,
        *,
        tiebreak_provider: Provider | None = None,
        ledger: SpendLedger | None = None,
    ) -> "Router":
        """Build a router from `config/` on disk, env overrides included."""
        return cls(
            load_config(config_dir),
            settings=load_classifier_settings(config_dir),
            tiebreak_provider=tiebreak_provider,
            ledger=ledger,
        )

    async def route(self, req: CompletionRequest, *, api_key: str | None = None) -> RouteResult:
        """Classify, apply the budget, and resolve the model chain.

        A caller that pins `req.model` keeps it as the chain head — the
        contract reads `model=None` as "router decides", so a set value is a
        request, not a hint — but the tier's chain still supplies the fallbacks.

        Raises `BudgetExceededError` when the key is over budget and
        `budgets.on_exceed` is `reject`.
        """
        classification = await self.classifier.classify(req)
        decision = await self.budget.apply(api_key, classification.complexity)
        chain = self.policy.chain_for(decision.tier).models
        if req.model:
            chain = [req.model, *(model for model in chain if model != req.model)]
        return RouteResult(
            tier=decision.tier,
            chain=chain,
            classification=classification,
            budget=decision,
        )

    async def record_spend(self, api_key: str | None, cost_usd: float) -> float:
        """Post-flight ledger write. Call once per completed request."""
        return await self.budget.record(api_key, cost_usd)
