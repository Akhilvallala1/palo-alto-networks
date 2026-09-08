"""Guard composition, threshold policy and verdict logging.

This module owns three things the individual guards deliberately do not:

1. **Composition.** `GuardChain` runs guards in order and threads redaction
   through them, so the injection guard inspects the text the vendor will
   actually receive rather than the raw one. The chain is itself a
   `conduit.contracts.Guard`, so the gateway sees one object.

2. **Threshold policy.** Thresholds come from `config/guards.yaml` and nowhere
   else. The chain re-applies the block threshold as defense in depth, but only
   to *unmitigated* risk: a verdict that carries `redacted_text` has already
   neutralised what it found, and PII risk and injection risk are not
   commensurable numbers to be maxed together into a block.

3. **Logging.** Every component verdict and the composed verdict are logged with
   guard name, allow/deny, risk score and categories — on the allow path too, so
   the record is a population, not just an incident list. `entity_map` and the
   inspected text are never logged: the map is the one thing in the system that
   must not cross the process boundary.

`fail_closed` is the enforcement switch, matching the comment in
`config/guards.yaml`: true means a verdict at or above threshold blocks; false
means the same verdict is recorded and the request proceeds, which is how you
run the guard in shadow mode before turning it on.
"""

import logging
from collections.abc import Sequence

from conduit.config import GuardsConfig
from conduit.contracts import Guard, GuardVerdict
from conduit.guards.injection import InjectionClassifier, InjectionGuard
from conduit.guards.pii import Analyzer, PIIGuard, find_placeholders, rehydrate

__all__ = [
    "GUARD_LOGGER_NAME",
    "GuardChain",
    "build_chain",
    "find_placeholders",
    "log_verdict",
    "rehydrate",
]

GUARD_LOGGER_NAME = "conduit.guards"

#: Fields copied onto the log record. `entity_map` is conspicuously absent.
_LOGGED_FIELDS = ("guard", "allowed", "risk_score", "categories", "entity_count", "redacted")


def log_verdict(
    guard_name: str,
    verdict: GuardVerdict,
    logger: logging.Logger | None = None,
) -> None:
    """Emit one structured record per verdict, allow or deny.

    Only the entity *count* is logged, never the map or the text it came from,
    so a log shipper can never become the exfiltration path that redaction was
    supposed to close.
    """
    log = logger if logger is not None else logging.getLogger(GUARD_LOGGER_NAME)
    log.log(
        logging.INFO if verdict.allowed else logging.WARNING,
        "guard verdict guard=%s allowed=%s risk_score=%.3f categories=%s",
        guard_name,
        verdict.allowed,
        verdict.risk_score,
        ",".join(verdict.categories) or "-",
        extra={
            "guard": guard_name,
            "allowed": verdict.allowed,
            "risk_score": round(verdict.risk_score, 4),
            "categories": list(verdict.categories),
            "entity_count": len(verdict.entity_map),
            "redacted": verdict.redacted_text is not None,
        },
    )


class GuardChain:
    """Runs a sequence of guards and composes one verdict from their results."""

    name = "chain"

    def __init__(
        self,
        guards: Sequence[Guard],
        config: GuardsConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._guards = tuple(guards)
        self._config = config if config is not None else GuardsConfig()
        self._logger = logger if logger is not None else logging.getLogger(GUARD_LOGGER_NAME)

    @property
    def guards(self) -> tuple[Guard, ...]:
        return self._guards

    @property
    def config(self) -> GuardsConfig:
        return self._config

    @property
    def risk_threshold(self) -> float:
        return self._config.injection.risk_threshold

    async def inspect(self, text: str) -> GuardVerdict:
        current = text
        risk = 0.0
        unmitigated_risk = 0.0
        blocked = False
        categories: list[str] = []
        entity_map: dict[str, str] = {}
        redacted = False

        for guard in self._guards:
            verdict = await self._run(guard, current)
            log_verdict(guard.name, verdict, self._logger)

            risk = max(risk, verdict.risk_score)
            if verdict.redacted_text is None:
                unmitigated_risk = max(unmitigated_risk, verdict.risk_score)
            else:
                current = verdict.redacted_text
                entity_map.update(verdict.entity_map)
                redacted = True
            for category in verdict.categories:
                if category not in categories:
                    categories.append(category)
            blocked = blocked or not verdict.allowed

        if unmitigated_risk >= self.risk_threshold:
            blocked = True

        allowed = not (blocked and self._config.fail_closed)
        if not allowed and not categories:
            categories = ["guard:threshold"]

        composed = GuardVerdict(
            allowed=allowed,
            risk_score=min(1.0, round(risk, 6)),
            categories=categories,
            redacted_text=current if redacted else None,
            entity_map=entity_map if redacted else {},
        )
        log_verdict(self.name, composed, self._logger)
        return composed

    async def _run(self, guard: Guard, text: str) -> GuardVerdict:
        """A guard that raises is a broken guard, and a broken guard denies.

        The contract says `inspect` never raises, but the chain is the last line
        between a bug in one guard and an unguarded request reaching a vendor.
        """
        try:
            return await guard.inspect(text)
        except Exception:  # a guard failure must not open the gate
            self._logger.exception("guard raised", extra={"guard": guard.name})
            return GuardVerdict(allowed=False, risk_score=1.0, categories=["guard:error"])


def build_chain(
    config: GuardsConfig | None = None,
    *,
    analyzer: Analyzer | None = None,
    classifier: InjectionClassifier | None = None,
    logger: logging.Logger | None = None,
) -> GuardChain:
    """The default ingress chain: PII redaction first, then injection scoring.

    PII runs first on purpose. Redacting before scoring means the injection
    layer — and, when enabled, the LLM classifier it may call — only ever sees
    placeholders, so turning on layer (c) does not quietly widen who sees raw
    customer data.
    """
    settings = config if config is not None else GuardsConfig()
    guards: list[Guard] = []
    if settings.pii.enabled:
        guards.append(PIIGuard(settings.pii, analyzer=analyzer))
    if settings.injection.enabled:
        guards.append(InjectionGuard(settings.injection, classifier=classifier))
    return GuardChain(guards, settings, logger=logger)
