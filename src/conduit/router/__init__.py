"""Complexity router and cost engine: request -> tier -> model chain.

Three moving parts, all importable from here:

* `classifier` — heuristic first, LLM tie-break only on low confidence.
* `policy` — tier to ordered model chain, plus the tier ladder and pricing.
* `cost` — per-key daily spend ledger and the `on_exceed` budget policy.

`Router` is the facade the gateway holds. The router never imports a concrete
provider: the tie-break model is reached through the `Provider` protocol.
"""

from .classifier import Classification, Classifier, ClassifierStats, LRUCache, prompt_hash
from .cost import (
    BUDGET_EXCEEDED_HEADER,
    BudgetDecision,
    BudgetEnforcer,
    BudgetExceededError,
    SpendLedger,
)
from .policy import TIER_LADDER, PolicyError, RoutingPolicy, TierChain, price_usd
from .service import Router, RouteResult
from .settings import ClassifierSettings, load_classifier_settings

__all__ = [
    "BUDGET_EXCEEDED_HEADER",
    "TIER_LADDER",
    "BudgetDecision",
    "BudgetEnforcer",
    "BudgetExceededError",
    "Classification",
    "Classifier",
    "ClassifierSettings",
    "ClassifierStats",
    "LRUCache",
    "PolicyError",
    "RouteResult",
    "Router",
    "RoutingPolicy",
    "SpendLedger",
    "TierChain",
    "load_classifier_settings",
    "price_usd",
    "prompt_hash",
]
