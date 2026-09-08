"""Unit tests for `conduit.guards.pii`: detection, redaction, rehydration."""

import pytest

from conduit.config import PIISettings
from conduit.contracts import Guard, GuardVerdict
from conduit.guards.pii import (
    CATEGORY_SUBTYPES,
    PLACEHOLDER_LABELS,
    Analyzer,
    EntitySpan,
    PIIGuard,
    PresidioAnalyzer,
    RegexAnalyzer,
    default_analyzer,
    find_placeholders,
    redact,
    rehydrate,
    resolve_overlaps,
)

# One unambiguous example per configured entity type. These are the eight the
# issue names, so a regression in any single recognizer fails its own test.
ENTITY_EXAMPLES: dict[str, tuple[str, str]] = {
    "EMAIL_ADDRESS": ("Send it to jane.doe@acmecorp.com today.", "jane.doe@acmecorp.com"),
    "PHONE_NUMBER": ("Their desk line is 415-555-0199 all week.", "415-555-0199"),
    "US_SSN": ("The form lists SSN 123-45-6789 in plain text.", "123-45-6789"),
    "CREDIT_CARD": ("Card on file is 4111 1111 1111 1111 now.", "4111 1111 1111 1111"),
    "IBAN_CODE": ("Wire to GB82WEST12345698765432 by Friday.", "GB82WEST12345698765432"),
    "PERSON": ("Our AE Jane Doe closed the expansion.", "Jane Doe"),
    "GTM_OPPORTUNITY_ID": ("Opportunity 006Ax000001BcDe is stalled.", "006Ax000001BcDe"),
    "GTM_QUOTE_NUMBER": ("Quote Q-2026-0442 needs sign-off.", "Q-2026-0442"),
}


@pytest.fixture
def guard() -> PIIGuard:
    return PIIGuard(PIISettings(), analyzer=RegexAnalyzer())


# --------------------------------------------------------------------------- #
# One test per entity type
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("entity_type", sorted(ENTITY_EXAMPLES), ids=sorted(ENTITY_EXAMPLES))
def test_each_entity_type_is_detected(guard: PIIGuard, entity_type: str) -> None:
    text, value = ENTITY_EXAMPLES[entity_type]
    spans = guard.detect(text)
    assert (entity_type, value) in {(s.entity_type, s.text) for s in spans}


@pytest.mark.parametrize("entity_type", sorted(ENTITY_EXAMPLES), ids=sorted(ENTITY_EXAMPLES))
async def test_each_entity_type_is_redacted_and_categorised(
    guard: PIIGuard, entity_type: str
) -> None:
    text, value = ENTITY_EXAMPLES[entity_type]
    verdict = await guard.inspect(text)
    assert f"pii:{CATEGORY_SUBTYPES[entity_type]}" in verdict.categories
    assert verdict.redacted_text is not None
    assert value not in verdict.redacted_text
    assert value in verdict.entity_map.values()
    placeholder = f"<{PLACEHOLDER_LABELS[entity_type]}_1>"
    assert placeholder in verdict.redacted_text


# --------------------------------------------------------------------------- #
# Validators and false-positive control
# --------------------------------------------------------------------------- #


def test_luhn_invalid_card_is_not_detected(guard: PIIGuard) -> None:
    assert guard.detect("Card 4111 1111 1111 1112 was rejected.") == []


def test_iban_with_bad_checksum_is_not_detected(guard: PIIGuard) -> None:
    assert guard.detect("Account GB00WEST12345698765432 is invalid.") == []


def test_bare_ten_digit_run_is_not_a_phone_number(guard: PIIGuard) -> None:
    assert guard.detect("Order 4455667788 shipped from Dallas.") == []


def test_nine_digits_need_ssn_context(guard: PIIGuard) -> None:
    assert guard.detect("Reference 234567891 is the tracking id.") == []
    assert guard.detect("Payroll shows social security number 234567891.") != []


def test_organisation_after_a_trigger_is_not_a_person(guard: PIIGuard) -> None:
    assert guard.detect("Please contact Support Team about the ticket.") == []
    assert guard.detect("Escalate to Legal Department for the redlines.") == []


def test_score_threshold_filters_low_confidence_spans() -> None:
    class _Weak:
        def analyze(self, text: str, entities: list[str] | tuple[str, ...]) -> list[EntitySpan]:
            return [EntitySpan("PERSON", 0, 4, 0.3, text[:4])]

    strict = PIIGuard(PIISettings(score_threshold=0.5), analyzer=_Weak())
    lenient = PIIGuard(PIISettings(score_threshold=0.2), analyzer=_Weak())
    assert strict.detect("Jane is here") == []
    assert len(lenient.detect("Jane is here")) == 1


# --------------------------------------------------------------------------- #
# Overlap, placeholders, rehydration
# --------------------------------------------------------------------------- #


def test_resolve_overlaps_keeps_the_more_confident_span() -> None:
    weak = EntitySpan("PHONE_NUMBER", 0, 12, 0.6, "123-45-6789 ")
    strong = EntitySpan("US_SSN", 0, 11, 0.9, "123-45-6789")
    assert resolve_overlaps([weak, strong]) == [strong]


def test_repeated_person_shares_one_placeholder(guard: PIIGuard) -> None:
    text = "Our AE Jane Doe asked twice; forward Jane Doe the signed copy."
    redacted, entity_map = redact(text, guard.detect(text))
    assert "Jane Doe" not in redacted
    assert list(entity_map) == ["<PERSON_1>"]
    assert redacted.count("<PERSON_1>") == 2


def test_redaction_round_trips_to_the_original(guard: PIIGuard) -> None:
    text = "Contact Jane Doe at jane.doe@acmecorp.com or 415-555-0199 about Q-2026-0442."
    redacted, entity_map = redact(text, guard.detect(text))
    assert redacted != text
    assert rehydrate(redacted, entity_map) == text


def test_rehydrate_prefers_the_longest_placeholder() -> None:
    entity_map = {"<PERSON_1>": "Ada", "<PERSON_10>": "Grace"}
    assert rehydrate("<PERSON_10> met <PERSON_1>", entity_map) == "Grace met Ada"


def test_rehydrate_leaves_unknown_placeholders_alone() -> None:
    assert rehydrate("hello <PERSON_99>", {"<PERSON_1>": "Ada"}) == "hello <PERSON_99>"


def test_rehydrate_with_an_empty_map_is_the_identity() -> None:
    assert rehydrate("nothing to do", {}) == "nothing to do"


def test_find_placeholders_reports_every_token() -> None:
    assert find_placeholders("<EMAIL_1> and <PERSON_2>") == ["<EMAIL_1>", "<PERSON_2>"]
    assert find_placeholders("no tokens here") == []


# --------------------------------------------------------------------------- #
# Guard behaviour
# --------------------------------------------------------------------------- #


async def test_clean_text_produces_an_allow_with_no_redaction(guard: PIIGuard) -> None:
    verdict = await guard.inspect("Summarize the renewal risk for the enterprise segment.")
    assert verdict.allowed
    assert verdict.risk_score == 0.0
    assert verdict.categories == []
    assert verdict.redacted_text is None
    assert verdict.entity_map == {}


async def test_detection_alone_does_not_block(guard: PIIGuard) -> None:
    """PII is mitigated by redaction, not by rejection - that is the design."""
    verdict = await guard.inspect("The form lists SSN 123-45-6789 in plain text.")
    assert verdict.allowed
    assert verdict.risk_score >= 0.9


async def test_disabling_redaction_blocks_instead(guard: PIIGuard) -> None:
    strict = PIIGuard(PIISettings(redact=False), analyzer=RegexAnalyzer())
    verdict = await strict.inspect("The form lists SSN 123-45-6789 in plain text.")
    assert not verdict.allowed
    assert verdict.categories == ["pii:ssn"]
    # Contract: redacted_text None implies an empty entity_map.
    assert verdict.redacted_text is None
    assert verdict.entity_map == {}


async def test_a_disabled_guard_allows_everything() -> None:
    off = PIIGuard(PIISettings(enabled=False), analyzer=RegexAnalyzer())
    verdict = await off.inspect("SSN 123-45-6789 and jane.doe@acmecorp.com.")
    assert verdict.allowed
    assert verdict.categories == []


async def test_an_exploding_analyzer_fails_closed_without_raising() -> None:
    class _Broken:
        def analyze(self, text: str, entities: list[str] | tuple[str, ...]) -> list[EntitySpan]:
            raise RuntimeError("model unavailable")

    verdict = await PIIGuard(PIISettings(), analyzer=_Broken()).inspect("anything")
    assert not verdict.allowed
    assert verdict.categories == ["pii:analyzer_error"]
    assert verdict.risk_score == 1.0


async def test_inspect_is_idempotent(guard: PIIGuard) -> None:
    text = "Contact Jane Doe at jane.doe@acmecorp.com about Q-2026-0442."
    assert await guard.inspect(text) == await guard.inspect(text)


async def test_risk_score_stays_within_the_contract_bounds(guard: PIIGuard) -> None:
    crowded = " ".join(text for text, _ in ENTITY_EXAMPLES.values())
    verdict = await guard.inspect(crowded)
    assert 0.0 <= verdict.risk_score <= 1.0


def test_guard_satisfies_the_frozen_protocol(guard: PIIGuard) -> None:
    checked: Guard = guard
    assert isinstance(checked, Guard)
    assert guard.name == "pii"


# --------------------------------------------------------------------------- #
# Analyzer selection
# --------------------------------------------------------------------------- #


def test_default_analyzer_returns_something_usable() -> None:
    analyzer = default_analyzer()
    assert isinstance(analyzer, Analyzer)
    assert analyzer.analyze("Quote Q-2026-0442 is open.", ("GTM_QUOTE_NUMBER",))


def test_presidio_adapter_maps_engine_results_and_merges_gtm_ids() -> None:
    class _Result:
        def __init__(self) -> None:
            self.entity_type = "PERSON"
            self.start = 0
            self.end = 8
            self.score = 0.85

    class _Engine:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def analyze(self, text: str, entities: list[str], language: str) -> list[_Result]:
            self.calls.append(entities)
            return [_Result()]

    engine = _Engine()
    adapter = PresidioAnalyzer(engine=engine)
    spans = adapter.analyze("Jane Doe owns quote Q-2026-0442.", ("PERSON", "GTM_QUOTE_NUMBER"))

    kinds = {(span.entity_type, span.text) for span in spans}
    assert ("PERSON", "Jane Doe") in kinds
    assert ("GTM_QUOTE_NUMBER", "Q-2026-0442") in kinds
    # GTM entities are ours, not Presidio's, so they are never forwarded to it.
    assert engine.calls == [["PERSON"]]


async def test_verdict_is_a_guard_verdict(guard: PIIGuard) -> None:
    assert isinstance(await guard.inspect("plain text"), GuardVerdict)


class TestSegmentedRedactionIsComposable:
    """Redacting a document in segments must not mint colliding placeholders.

    The gateway inspects several `Message` bodies per request. Before `redact()`
    accepted a continuation map, each call restarted numbering, so two different
    people both became `<PERSON_1>`; merging the maps rehydrated one person's name
    into the other's sentence. In a lead-to-cash flow that is a quote addressed to
    the wrong customer, which is worse than leaking the name.
    """

    def _redact_all(self, segments: list[str]) -> tuple[list[str], dict[str, str]]:
        analyzer = RegexAnalyzer()
        entity_map: dict[str, str] = {}
        out: list[str] = []
        for segment in segments:
            text, entity_map = redact(
                segment, analyzer.analyze(segment, ("EMAIL_ADDRESS",)), entity_map
            )
            out.append(text)
        return out, entity_map

    def test_two_people_in_two_segments_get_distinct_placeholders(self) -> None:
        texts, entity_map = self._redact_all(
            ["Contact is alice@northwind.example.", "Escalate to bob@contoso.example."]
        )
        first = find_placeholders(texts[0])
        second = find_placeholders(texts[1])
        assert first and second
        assert not set(first) & set(second), (
            f"segments collided: {first} vs {second} — the same placeholder now maps "
            "to two different originals and rehydration returns the wrong value"
        )
        assert len(set(entity_map.values())) == len(entity_map)

    def test_the_same_original_reuses_its_placeholder_across_segments(self) -> None:
        texts, entity_map = self._redact_all(
            ["Quote for alice@northwind.example.", "Approver: alice@northwind.example."]
        )
        assert find_placeholders(texts[0]) == find_placeholders(texts[1])
        assert len(entity_map) == 1

    def test_round_trip_through_segments_restores_every_original(self) -> None:
        segments = [
            "Contact is alice@northwind.example.",
            "Escalate to bob@contoso.example.",
            "Cc alice@northwind.example. again.",
        ]
        texts, entity_map = self._redact_all(segments)
        assert [rehydrate(t, entity_map) for t in texts] == segments
        assert not any(find_placeholders(rehydrate(t, entity_map)) for t in texts)

    def test_without_a_continuation_map_the_collision_is_still_reproducible(self) -> None:
        """Documents why the parameter exists: the old call shape still collides."""
        analyzer = RegexAnalyzer()
        a, map_a = redact(
            "Contact is alice@northwind.example.",
            analyzer.analyze("Contact is alice@northwind.example.", ("EMAIL_ADDRESS",)),
        )
        b, map_b = redact(
            "Escalate to bob@contoso.example.",
            analyzer.analyze("Escalate to bob@contoso.example.", ("EMAIL_ADDRESS",)),
        )
        assert set(find_placeholders(a)) & set(find_placeholders(b))
        assert map_a != map_b
