"""Reversible PII detection, redaction and egress rehydration.

Redaction here is deliberately *reversible*. A GTM workflow that drops the
customer's name produces a quote addressed to nobody, so dropping PII is easier
and useless. Instead every detected span is swapped for a stable placeholder,
the mapping is kept in `GuardVerdict.entity_map`, and the egress path calls
`rehydrate` to put the originals back. The net effect is that the vendor never
sees raw PII and the caller never sees a placeholder.

Detection has two backends behind one `Analyzer` protocol:

- `RegexAnalyzer` — the default. Deterministic, in-process, zero network, zero
  extra dependencies. Validators (Luhn, IBAN mod-97) and context triggers do the
  work that a statistical model would otherwise do, which keeps the
  false-positive rate measurable rather than model-version-dependent.
- `PresidioAnalyzer` — used when `presidio-analyzer` is installed. Adds
  NER-backed `PERSON` recall on free text. It is an optional extra, imported
  lazily, and `default_analyzer()` silently falls back to `RegexAnalyzer` when
  it is absent so that CI and the hot path never depend on a model download.

`entity_map` is process-local by construction: it lives on the verdict, it is
never logged (see `conduit.guards.base.log_verdict`), and nothing in this module
serialises it.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from conduit.config import PIISettings
from conduit.contracts import GuardVerdict

__all__ = [
    "CATEGORY_SUBTYPES",
    "PLACEHOLDER_LABELS",
    "PLACEHOLDER_RE",
    "Analyzer",
    "EntitySpan",
    "PIIGuard",
    "PresidioAnalyzer",
    "RegexAnalyzer",
    "default_analyzer",
    "find_placeholders",
    "redact",
    "rehydrate",
    "resolve_overlaps",
]


# --------------------------------------------------------------------------- #
# Entity vocabulary
# --------------------------------------------------------------------------- #

#: Entity type -> the label used inside a placeholder token, e.g. `<EMAIL_1>`.
PLACEHOLDER_LABELS: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "GTM_OPPORTUNITY_ID": "OPPORTUNITY_ID",
    "GTM_QUOTE_NUMBER": "QUOTE_NUMBER",
}

#: Entity type -> the `pii:<subtype>` suffix reported in `GuardVerdict.categories`.
CATEGORY_SUBTYPES: dict[str, str] = {
    "PERSON": "person",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "US_SSN": "ssn",
    "CREDIT_CARD": "credit_card",
    "IBAN_CODE": "iban",
    "GTM_OPPORTUNITY_ID": "opportunity_id",
    "GTM_QUOTE_NUMBER": "quote_number",
}

#: How much exposure each entity type represents, used to derive `risk_score`.
SENSITIVITY: dict[str, float] = {
    "US_SSN": 1.0,
    "CREDIT_CARD": 1.0,
    "IBAN_CODE": 0.9,
    "EMAIL_ADDRESS": 0.6,
    "PHONE_NUMBER": 0.6,
    "PERSON": 0.5,
    "GTM_OPPORTUNITY_ID": 0.4,
    "GTM_QUOTE_NUMBER": 0.4,
}

#: Matches any placeholder this module can emit. Used to assert that no
#: placeholder survives to the caller.
PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]*?)_(\d+)>")


@dataclass(frozen=True)
class EntitySpan:
    """One detected PII occurrence, as a half-open `[start, end)` slice."""

    entity_type: str
    start: int
    end: int
    score: float
    text: str


# --------------------------------------------------------------------------- #
# Regex recognizers
# --------------------------------------------------------------------------- #

EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)

# A separator is mandatory: bare 10-digit runs are far more often order numbers
# than phone numbers, and the false positives are what the corpus measures.
PHONE_RE = re.compile(
    r"(?<![\w-])(?:\+?1[ .\-]?)?(?:\(\d{3}\)\s?|\d{3}[ .\-])\d{3}[ .\-]\d{4}(?![\w-])"
    r"|(?<![\w-])\+\d{1,3}[ .\-]\d{1,4}[ .\-]\d{2,4}[ .\-]\d{2,4}(?![\w-])"
)

SSN_RE = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")
# Nine bare digits are only an SSN when something nearby says so.
SSN_CONTEXT_RE = re.compile(r"(?i:\b(?:ssn|social security(?:\s+number)?)\b)\D{0,15}(\d{9})(?!\d)")

CREDIT_CARD_RE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
IBAN_RE = re.compile(r"(?<![\w-])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}[ ]?[A-Z0-9]{1,4}(?![\w-])")

OPPORTUNITY_RE = re.compile(
    r"(?<![\w-])(?:006[A-Za-z0-9]{12}(?:[A-Za-z0-9]{3})?|OPP-\d{4,10})(?![\w-])"
)
QUOTE_RE = re.compile(r"(?<![\w-])(?:Q-\d{4}-\d{3,6}|QUO-\d{5,10}|QT-?\d{6,10})(?![\w-])")

# A person is a capitalised 2-3 token run. Requiring a trigger word in front of
# it is what keeps "Acme Global Networks" out of the results. The inner
# `[A-Z][a-z]+` tail is what lets "O'Sullivan" and "McDonald" through while
# still rejecting all-caps tokens like "ACME".
_TOKEN = r"[A-Z][a-z\u2019']{1,20}(?:[A-Z][a-z]{1,20})?"
_NAME = rf"{_TOKEN}(?:\s+(?:[A-Z]\.|{_TOKEN})){{1,2}}"
_SHORT_NAME = rf"{_TOKEN}(?:\s+(?:[A-Z]\.\s*)?{_TOKEN})?"

HONORIFIC_RE = re.compile(rf"(?i:\b(?:mr|mrs|ms|miss|dr|prof)\b)\.?\s+({_SHORT_NAME})")
PERSON_CONTEXT_RE = re.compile(
    r"(?i:\b(?:contact|attn|attention|owner|owned by|rep|ae|account executive|csm|signed by"
    r"|approved by|rejected by|prepared by|prepared for|assigned to|escalated to|escalate to"
    r"|reach out to|introduced to|spoke with|met with|name is|named|cc|reply to|billed to"
    r"|shipped to|authored by|requested by|submitted by|champion|counter-?signed by)\b)"
    rf"\W{{0,4}}({_NAME})"
)
# Document nouns carry an implicit person reference: "onboarding packet for X",
# "renewal note to X". Narrow enough that "contract for North America" is still
# rejected by the token stoplist.
PERSON_DOCUMENT_RE = re.compile(
    r"(?i:\b(?:packet|record|profile|file|form|quote|contract|agreement|invoice|report|letter"
    r"|note|summary|memo|order|proposal)\s+(?:is\s+)?(?:for|to|from)\b)"
    rf"\W{{0,4}}({_NAME})"
)
SIGNOFF_RE = re.compile(
    rf"(?i:\b(?:regards|sincerely|thanks|thank you|best|cheers|warmly)\b),?\s*\n?\s*({_SHORT_NAME})"
)

#: Tokens that look like a surname but mark an organisation or a calendar word.
#: A word list this long reads as prose, not as a hundred quoted strings, so it
#: is written as one and split at import time.
_NOT_A_NAME = frozenset(
    """
    corp corporation inc llc ltd plc gmbh co company systems solutions
    technologies technology industries group partners holdings networks labs
    software hardware health bank capital media global services team support
    sales legal finance security cloud data dynamics analytics enterprises
    ventures consulting associates international division department region unit
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october
    november december quarter
    north south east west america americas emea apac europe asia
    federal state city county the our your their please note
    today tomorrow yesterday
    renewal quote invoice opportunity account contract discount pricing platform
    edition
    """.split()  # noqa: SIM905 - a word list this long reads better as prose
)


def _luhn_ok(digits: str) -> bool:
    """Standard mod-10 checksum. Rejects the repdigit strings regexes love."""
    if len(set(digits)) == 1:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    """ISO 13616 mod-97 check: the rearranged, letter-expanded value must be 1."""
    compact = candidate.replace(" ", "")
    if not 15 <= len(compact) <= 34:
        return False
    rotated = compact[4:] + compact[:4]
    expanded = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rotated)
    if not expanded.isdigit():
        return False
    return int(expanded) % 97 == 1


def _name_is_plausible(candidate: str) -> bool:
    tokens = [token.strip(".,;:") for token in candidate.split()]
    if not tokens:
        return False
    return all(token.lower().strip("'\u2019") not in _NOT_A_NAME for token in tokens)


def _scan(pattern: re.Pattern[str], text: str, entity_type: str, score: float) -> list[EntitySpan]:
    return [
        EntitySpan(entity_type, m.start(), m.end(), score, m.group(0))
        for m in pattern.finditer(text)
    ]


def _scan_group(
    pattern: re.Pattern[str], text: str, entity_type: str, score: float
) -> list[EntitySpan]:
    """Like `_scan` but keeps only capture group 1, so the trigger word is not redacted."""
    spans: list[EntitySpan] = []
    for match in pattern.finditer(text):
        value = match.group(1)
        if entity_type == "PERSON" and not _name_is_plausible(value):
            continue
        spans.append(EntitySpan(entity_type, match.start(1), match.end(1), score, value))
    return spans


def _scan_validated(
    pattern: re.Pattern[str],
    text: str,
    entity_type: str,
    score: float,
    validator: Callable[[str], bool],
) -> list[EntitySpan]:
    """`_scan` gated by a checksum. A regex proposes; the validator disposes."""
    spans: list[EntitySpan] = []
    for match in pattern.finditer(text):
        value = match.group(0)
        if not validator(value):
            continue
        spans.append(EntitySpan(entity_type, match.start(), match.end(), score, value))
    return spans


def _card_ok(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


def _propagate(found: Sequence[EntitySpan], text: str) -> list[EntitySpan]:
    """Every other literal occurrence of an already-confirmed name.

    A name only needs one trigger to be recognised ("contact Jane Doe"); the
    bare repetitions later in the same text are the same person and must be
    redacted too, or the vendor payload leaks the PII the first hit caught.
    """
    extra: list[EntitySpan] = []
    for value in {span.text for span in found}:
        for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", text):
            extra.append(EntitySpan("PERSON", match.start(), match.end(), 0.7, value))
    return extra


# --------------------------------------------------------------------------- #
# Analyzers
# --------------------------------------------------------------------------- #


@runtime_checkable
class Analyzer(Protocol):
    """A PII detector. Implementations must be pure and must not do network I/O
    unless they say so; `PIIGuard` calls this synchronously on the hot path."""

    def analyze(self, text: str, entities: Sequence[str]) -> list[EntitySpan]: ...


class RegexAnalyzer:
    """Deterministic, dependency-free detection with checksum validation.

    `PERSON` is context-triggered rather than statistical: a capitalised name run
    only counts when an honorific, a GTM role word ("our AE", "signed by") or a
    sign-off precedes it. That trades a little recall on bare names for a false
    positive rate that does not move when a model version changes.
    """

    def analyze(self, text: str, entities: Sequence[str]) -> list[EntitySpan]:
        wanted = set(entities)
        spans: list[EntitySpan] = []

        if "EMAIL_ADDRESS" in wanted:
            spans += _scan(EMAIL_RE, text, "EMAIL_ADDRESS", 0.95)
        if "US_SSN" in wanted:
            spans += _scan(SSN_RE, text, "US_SSN", 0.9)
            spans += _scan_group(SSN_CONTEXT_RE, text, "US_SSN", 0.85)
        if "CREDIT_CARD" in wanted:
            spans += _scan_validated(CREDIT_CARD_RE, text, "CREDIT_CARD", 0.95, _card_ok)
        if "IBAN_CODE" in wanted:
            spans += _scan_validated(IBAN_RE, text, "IBAN_CODE", 0.95, _iban_ok)
        if "PHONE_NUMBER" in wanted:
            spans += _scan(PHONE_RE, text, "PHONE_NUMBER", 0.85)
        if "GTM_OPPORTUNITY_ID" in wanted:
            spans += _scan(OPPORTUNITY_RE, text, "GTM_OPPORTUNITY_ID", 0.9)
        if "GTM_QUOTE_NUMBER" in wanted:
            spans += _scan(QUOTE_RE, text, "GTM_QUOTE_NUMBER", 0.9)
        if "PERSON" in wanted:
            people = _scan_group(HONORIFIC_RE, text, "PERSON", 0.8)
            people += _scan_group(PERSON_CONTEXT_RE, text, "PERSON", 0.75)
            people += _scan_group(PERSON_DOCUMENT_RE, text, "PERSON", 0.75)
            people += _scan_group(SIGNOFF_RE, text, "PERSON", 0.7)
            spans += people + _propagate(people, text)

        return spans


class PresidioResult(Protocol):
    """The shape of one `presidio_analyzer.RecognizerResult`.

    Declared structurally rather than imported so that this module type-checks
    with the optional extra absent, which is the state CI runs in.
    """

    entity_type: str
    start: int
    end: int
    score: float


class PresidioEngine(Protocol):
    def analyze(
        self, text: str, entities: list[str], language: str
    ) -> Sequence[PresidioResult]: ...


class PresidioAnalyzer:
    """Adapter over `presidio_analyzer.AnalyzerEngine`, plus the GTM regexes.

    Presidio has no notion of an opportunity id or a quote number, so the regex
    analyzer still runs for those and its spans are merged in. Constructing this
    loads a spaCy model, so it is built once and reused; `default_analyzer()`
    never constructs it implicitly.
    """

    def __init__(self, engine: PresidioEngine | None = None, language: str = "en") -> None:
        self._language = language
        self._regex = RegexAnalyzer()
        self._gtm_only = tuple(name for name in PLACEHOLDER_LABELS if name.startswith("GTM_"))
        if engine is None:
            engine = _load_presidio_engine()
        self._engine = engine

    def analyze(self, text: str, entities: Sequence[str]) -> list[EntitySpan]:
        presidio_entities = [name for name in entities if not name.startswith("GTM_")]
        spans: list[EntitySpan] = []
        if presidio_entities:
            results = self._engine.analyze(
                text=text, entities=presidio_entities, language=self._language
            )
            spans += [
                EntitySpan(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    text=text[result.start : result.end],
                )
                for result in results
            ]
        gtm = [name for name in entities if name in self._gtm_only]
        if gtm:
            spans += self._regex.analyze(text, gtm)
        return spans


def _load_presidio_engine() -> PresidioEngine:
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via default_analyzer
        raise ImportError(
            "presidio-analyzer is not installed; PIIGuard falls back to RegexAnalyzer"
        ) from exc
    engine: PresidioEngine = AnalyzerEngine()
    return engine


def default_analyzer() -> Analyzer:
    """`PresidioAnalyzer` when the optional extra is installed, else regex only.

    Falling back rather than raising is deliberate: the guard must run in-process
    with no network and no model download in CI, and a missing optional extra is
    a capability difference, not an error.
    """
    try:
        return PresidioAnalyzer()
    except Exception:  # any import/model-load failure means fall back
        return RegexAnalyzer()


# --------------------------------------------------------------------------- #
# Redaction / rehydration
# --------------------------------------------------------------------------- #


def resolve_overlaps(spans: Iterable[EntitySpan]) -> list[EntitySpan]:
    """Keep the most confident, then longest, span of each overlapping cluster.

    Recognizers deliberately overlap — a credit card and a phone number are both
    "digits with separators" — so a winner has to be picked before any text is
    rewritten, otherwise offsets shift under each other.
    """
    ranked = sorted(spans, key=lambda s: (-s.score, -(s.end - s.start), s.start, s.entity_type))
    kept: list[EntitySpan] = []
    for span in ranked:
        if any(span.start < other.end and other.start < span.end for other in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def redact(text: str, spans: Sequence[EntitySpan]) -> tuple[str, dict[str, str]]:
    """Replace each span with `<LABEL_n>`, returning the new text and the map.

    Identical originals of the same type share one placeholder, so a name that
    appears three times stays one entity to the model and rehydrates uniformly.
    """
    ordered = resolve_overlaps(spans)
    counters: dict[str, int] = {}
    assigned: dict[tuple[str, str], str] = {}
    entity_map: dict[str, str] = {}
    pieces: list[str] = []
    cursor = 0

    for span in ordered:
        key = (span.entity_type, span.text)
        placeholder = assigned.get(key)
        if placeholder is None:
            label = PLACEHOLDER_LABELS.get(span.entity_type, span.entity_type)
            counters[label] = counters.get(label, 0) + 1
            placeholder = f"<{label}_{counters[label]}>"
            assigned[key] = placeholder
            entity_map[placeholder] = span.text
        pieces.append(text[cursor : span.start])
        pieces.append(placeholder)
        cursor = span.end

    pieces.append(text[cursor:])
    return "".join(pieces), entity_map


def rehydrate(text: str, entity_map: dict[str, str]) -> str:
    """Egress helper: put the originals back before the caller sees the text.

    Longest placeholder first, so `<PERSON_1>` cannot eat the prefix of
    `<PERSON_10>`. Unknown placeholders are left untouched rather than blanked —
    a model that invented `<PERSON_99>` is a bug to surface, not to hide.
    """
    if not entity_map:
        return text
    for placeholder in sorted(entity_map, key=len, reverse=True):
        text = text.replace(placeholder, entity_map[placeholder])
    return text


def find_placeholders(text: str) -> list[str]:
    """Every placeholder-shaped token in `text`. Used by the round-trip assertions."""
    return [match.group(0) for match in PLACEHOLDER_RE.finditer(text)]


# --------------------------------------------------------------------------- #
# Guard
# --------------------------------------------------------------------------- #


class PIIGuard:
    """`conduit.contracts.Guard` implementation for reversible PII redaction.

    Detected PII is *mitigated*, not rejected: the verdict stays `allowed=True`
    and carries `redacted_text` plus `entity_map`. The one case that blocks is
    `redact: false` in `config/guards.yaml` — detection with no mitigation
    available is the only honest fail-closed answer there.
    """

    name = "pii"

    def __init__(
        self,
        settings: PIISettings | None = None,
        analyzer: Analyzer | None = None,
    ) -> None:
        self._settings = settings or PIISettings()
        self._analyzer = analyzer if analyzer is not None else default_analyzer()
        self._entities = tuple(self._settings.entities or PLACEHOLDER_LABELS.keys())

    @property
    def analyzer(self) -> Analyzer:
        return self._analyzer

    def detect(self, text: str) -> list[EntitySpan]:
        """Confidence-filtered, overlap-resolved spans. Pure; safe to call twice."""
        spans = [
            span
            for span in self._analyzer.analyze(text, self._entities)
            if span.score >= self._settings.score_threshold and span.entity_type in self._entities
        ]
        return resolve_overlaps(spans)

    async def inspect(self, text: str) -> GuardVerdict:
        if not self._settings.enabled:
            return GuardVerdict(allowed=True, risk_score=0.0, categories=[])
        try:
            spans = self.detect(text)
        except Exception:  # the Guard contract forbids raising
            return GuardVerdict(allowed=False, risk_score=1.0, categories=["pii:analyzer_error"])

        if not spans:
            return GuardVerdict(allowed=True, risk_score=0.0, categories=[])

        categories = _categories_for(spans)
        risk = _risk_for(spans)

        if not self._settings.redact:
            # No mitigation configured, so the only safe verdict is a block. The
            # contract requires an empty entity_map whenever redacted_text is None.
            return GuardVerdict(allowed=False, risk_score=risk, categories=categories)

        redacted_text, entity_map = redact(text, spans)
        return GuardVerdict(
            allowed=True,
            risk_score=risk,
            categories=categories,
            redacted_text=redacted_text,
            entity_map=entity_map,
        )


def _categories_for(spans: Sequence[EntitySpan]) -> list[str]:
    seen: list[str] = []
    for span in spans:
        subtype = CATEGORY_SUBTYPES.get(span.entity_type, span.entity_type.lower())
        category = f"pii:{subtype}"
        if category not in seen:
            seen.append(category)
    return sorted(seen)


def _risk_for(spans: Sequence[EntitySpan]) -> float:
    """Sensitivity of the worst entity, nudged up by volume, clamped to 1.0."""
    worst = max(SENSITIVITY.get(span.entity_type, 0.5) for span in spans)
    return min(1.0, round(worst + 0.05 * (len(spans) - 1), 6))
