"""Security guards implementing `conduit.contracts.Guard`: PII and injection.

Ingress::

    from conduit.config import load_guards
    from conduit.guards import build_chain

    chain = build_chain(load_guards())
    verdict = await chain.inspect(user_text)
    if not verdict.allowed:
        raise HTTPException(400, detail=verdict.categories)
    vendor_payload = verdict.redacted_text or user_text

Egress::

    from conduit.guards import rehydrate

    caller_text = rehydrate(vendor_response.text, verdict.entity_map)

`verdict.entity_map` is process-local. Hold it for the duration of the request,
pass it to `rehydrate`, and never log, persist or serialise it.
"""

from conduit.guards.base import (
    GUARD_LOGGER_NAME,
    GuardChain,
    build_chain,
    find_placeholders,
    log_verdict,
    rehydrate,
)
from conduit.guards.injection import (
    FAMILIES,
    InjectionClassifier,
    InjectionGuard,
    InjectionSignals,
)
from conduit.guards.pii import (
    Analyzer,
    EntitySpan,
    PIIGuard,
    PresidioAnalyzer,
    RegexAnalyzer,
    default_analyzer,
)

__all__ = [
    "FAMILIES",
    "GUARD_LOGGER_NAME",
    "Analyzer",
    "EntitySpan",
    "GuardChain",
    "InjectionClassifier",
    "InjectionGuard",
    "InjectionSignals",
    "PIIGuard",
    "PresidioAnalyzer",
    "RegexAnalyzer",
    "build_chain",
    "default_analyzer",
    "find_placeholders",
    "log_verdict",
    "rehydrate",
]
