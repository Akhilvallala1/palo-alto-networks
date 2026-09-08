"""Eval suite: guard precision and recall against the committed corpora.

This is the suite the acceptance criteria are written against, so it reports
measured rates rather than spot-checking examples:

- PII recall >= 95% and false-positive rate <= 5% on `pii_corpus.jsonl`
- injection detection >= 90% in *every* one of the eight families, and <= 10%
  false positives on the benign lookalikes

Recall is counted per entity occurrence, not per row, so a row with four
entities cannot be scored as a single hit. The false-positive rate is counted
per row, which is the unit an operator experiences: one spurious flag ruins one
request regardless of how many spans caused it.

The measurement runs against `RegexAnalyzer` explicitly. Using
`default_analyzer()` would make the published numbers depend on whether an
optional extra happened to be installed on the machine that ran them.
"""

from collections import defaultdict

import pytest

from conduit.config import InjectionSettings, PIISettings
from conduit.guards.injection import FAMILIES, InjectionGuard
from conduit.guards.pii import PIIGuard, RegexAnalyzer
from guards_corpus import attacks, benign, injection_cases, pii_cases

PII_MIN_RECALL = 0.95
PII_MAX_FP_RATE = 0.05
INJECTION_MIN_DETECTION = 0.90
INJECTION_MAX_FP_RATE = 0.10


@pytest.fixture(scope="module")
def pii_guard() -> PIIGuard:
    return PIIGuard(PIISettings(), analyzer=RegexAnalyzer())


@pytest.fixture(scope="module")
def injection_guard() -> InjectionGuard:
    return InjectionGuard(InjectionSettings())


# --------------------------------------------------------------------------- #
# Corpus shape - the numbers below are only meaningful if the corpus is
# --------------------------------------------------------------------------- #


def test_pii_corpus_covers_every_configured_entity_type() -> None:
    covered = {entity for case in pii_cases() for entity, _ in case.entities}
    assert covered == set(PIISettings(entities=[]).entities) | covered  # no unknown types
    assert covered == {
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IBAN_CODE",
        "GTM_OPPORTUNITY_ID",
        "GTM_QUOTE_NUMBER",
    }


def test_pii_corpus_has_enough_negatives_to_measure_false_positives() -> None:
    negatives = [case for case in pii_cases() if case.is_negative]
    assert len(negatives) >= 30
    assert len(pii_cases()) - len(negatives) >= 50


def test_injection_corpus_is_stratified_over_all_eight_families() -> None:
    by_family: dict[str, int] = defaultdict(int)
    for case in attacks():
        by_family[case.family] += 1
    assert set(by_family) == set(FAMILIES)
    assert min(by_family.values()) >= 5, by_family
    assert len(attacks()) >= 60
    assert len(benign()) >= 40


def test_injection_corpus_ids_are_unique() -> None:
    ids = [case.id for case in injection_cases()]
    assert len(set(ids)) == len(ids)


def test_pii_corpus_values_are_exact_substrings_of_their_text() -> None:
    for case in pii_cases():
        for _, value in case.entities:
            assert value in case.text, case.id


# --------------------------------------------------------------------------- #
# PII precision / recall
# --------------------------------------------------------------------------- #


def test_pii_recall_meets_the_acceptance_threshold(pii_guard: PIIGuard) -> None:
    hits = 0
    total = 0
    missed: list[str] = []
    for case in pii_cases():
        found = {(span.entity_type, span.text) for span in pii_guard.detect(case.text)}
        for entity in case.entities:
            total += 1
            if entity in found:
                hits += 1
            else:
                missed.append(f"{case.id}: {entity[0]}={entity[1]!r}")

    recall = hits / total
    assert recall >= PII_MIN_RECALL, f"recall {recall:.3f} ({hits}/{total}); missed: {missed}"


def test_pii_false_positive_rate_meets_the_acceptance_threshold(pii_guard: PIIGuard) -> None:
    negatives = [case for case in pii_cases() if case.is_negative]
    flagged = [
        (case.id, [(s.entity_type, s.text) for s in pii_guard.detect(case.text)])
        for case in negatives
        if pii_guard.detect(case.text)
    ]

    rate = len(flagged) / len(negatives)
    assert rate <= PII_MAX_FP_RATE, (
        f"FP rate {rate:.3f} ({len(flagged)}/{len(negatives)}): {flagged}"
    )


@pytest.mark.parametrize(
    "entity_type",
    [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IBAN_CODE",
        "GTM_OPPORTUNITY_ID",
        "GTM_QUOTE_NUMBER",
    ],
)
def test_per_entity_recall_meets_the_acceptance_threshold(
    pii_guard: PIIGuard, entity_type: str
) -> None:
    """An aggregate can hide one dead recognizer; this cannot."""
    hits = 0
    total = 0
    for case in pii_cases():
        found = {(span.entity_type, span.text) for span in pii_guard.detect(case.text)}
        for entity in case.entities:
            if entity[0] != entity_type:
                continue
            total += 1
            hits += entity in found

    assert total >= 4, f"{entity_type} is under-represented in the corpus"
    assert hits / total >= PII_MIN_RECALL, f"{entity_type} recall {hits}/{total}"


# --------------------------------------------------------------------------- #
# Injection precision / recall
# --------------------------------------------------------------------------- #


async def test_injection_detection_meets_the_acceptance_threshold(
    injection_guard: InjectionGuard,
) -> None:
    missed: list[str] = []
    hits = 0
    for case in attacks():
        verdict = await injection_guard.inspect(case.text)
        if verdict.allowed:
            missed.append(f"{case.id} ({case.family}) score={verdict.risk_score:.2f}")
        else:
            hits += 1

    rate = hits / len(attacks())
    assert rate >= INJECTION_MIN_DETECTION, f"detection {rate:.3f}; missed: {missed}"


@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
async def test_per_family_detection_meets_the_acceptance_threshold(
    injection_guard: InjectionGuard, family: str
) -> None:
    """AC-3 says >=90% across all eight families, not >=90% on average."""
    cases = [case for case in attacks() if case.family == family]
    assert cases, f"no attacks for {family}"

    hits = 0
    for case in cases:
        verdict = await injection_guard.inspect(case.text)
        hits += not verdict.allowed

    rate = hits / len(cases)
    assert rate >= INJECTION_MIN_DETECTION, f"{family} detection {hits}/{len(cases)}"


async def test_injection_false_positive_rate_meets_the_acceptance_threshold(
    injection_guard: InjectionGuard,
) -> None:
    flagged: list[str] = []
    for case in benign():
        verdict = await injection_guard.inspect(case.text)
        if not verdict.allowed:
            flagged.append(f"{case.id} ({case.family}) {verdict.categories}")

    rate = len(flagged) / len(benign())
    assert rate <= INJECTION_MAX_FP_RATE, f"FP rate {rate:.3f}: {flagged}"


async def test_every_verdict_in_the_corpus_carries_a_score_and_categories(
    injection_guard: InjectionGuard,
) -> None:
    """AC-5: allow or deny, a verdict is always explainable."""
    for case in injection_cases():
        verdict = await injection_guard.inspect(case.text)
        assert 0.0 <= verdict.risk_score <= 1.0, case.id
        if not verdict.allowed:
            assert verdict.categories, case.id
