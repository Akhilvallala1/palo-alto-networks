"""Unit tests for the two-stage complexity classifier."""

import pytest

from conduit.contracts import CompletionRequest, Complexity, Message, Provider
from conduit.router.classifier import (
    Classifier,
    LRUCache,
    _imperative_verbs,
    estimate_tokens,
    prompt_hash,
)
from conduit.router.settings import ClassifierSettings
from router_fakes import FakeProvider


def request(text: str, **metadata: str) -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content=text)], metadata=metadata)


# --------------------------------------------------------------------------- #
# Stage 1 features
# --------------------------------------------------------------------------- #


def test_trivial_verb_and_structured_marker_route_to_trivial() -> None:
    verdict = Classifier().heuristic(
        request("Extract the opportunity id from this note. Return JSON only.")
    )
    assert verdict.complexity is Complexity.TRIVIAL
    assert verdict.source == "heuristic"
    assert any(feature.startswith("verbs:trivial") for feature in verdict.features)
    assert any(feature.startswith("structured") for feature in verdict.features)


def test_drafting_verb_routes_to_standard() -> None:
    verdict = Classifier().heuristic(
        request("Draft a follow-up email to the champion recapping our pricing conversation.")
    )
    assert verdict.complexity is Complexity.STANDARD


def test_code_fence_routes_to_complex() -> None:
    verdict = Classifier().heuristic(
        request("What is wrong here?\n\n```python\ndef total(x):\n    return x * 1.1\n```")
    )
    assert verdict.complexity is Complexity.COMPLEX
    assert "code_fence" in verdict.features


def test_workflow_hint_dominates_the_text() -> None:
    # Text alone reads trivial; the caller's workflow tag says otherwise, and it
    # must win outright rather than merely nudge.
    unhinted = Classifier().heuristic(request("Extract the discount."))
    hinted = Classifier().heuristic(request("Extract the discount.", workflow="discount_analyst"))
    assert unhinted.complexity is Complexity.TRIVIAL
    assert hinted.complexity is Complexity.COMPLEX
    assert "workflow_hint:discount_analyst" in hinted.features
    # High enough to settle it without paying for a tie-break.
    assert hinted.confidence >= ClassifierSettings().confidence_threshold


def test_unknown_workflow_hint_is_ignored() -> None:
    verdict = Classifier().heuristic(request("Extract the discount.", workflow="not_a_workflow"))
    assert verdict.complexity is Complexity.TRIVIAL
    assert not any(feature.startswith("workflow_hint") for feature in verdict.features)


def test_three_or_more_questions_signal_complex() -> None:
    verdict = Classifier().heuristic(
        request(
            "Should we escalate this deal? What are the risks of each path? "
            "Which policy sections govern it?"
        )
    )
    assert verdict.complexity is Complexity.COMPLEX
    assert any(feature.startswith("questions:") for feature in verdict.features)


def test_arithmetic_cue_needs_both_digits_and_a_money_word() -> None:
    with_digits = Classifier().heuristic(request("Work out the total margin on 12 units at 18%."))
    without_digits = Classifier().heuristic(request("Tell me about margin in general terms."))
    assert "arithmetic" in with_digits.features
    assert "arithmetic" not in without_digits.features


def test_hard_verb_beats_subordinate_drafting_verb() -> None:
    verdict = Classifier().heuristic(
        request("Analyze this discount request against policy and explain your decision.")
    )
    assert verdict.complexity is Complexity.COMPLEX
    assert "verbs_subordinate:standard" in verdict.features


def test_verb_as_a_noun_does_not_outrank_the_imperative() -> None:
    verdict = Classifier().heuristic(
        request("Summarize these pipeline notes for the weekly forecast call.")
    )
    assert verdict.complexity is Complexity.STANDARD


def test_imperative_detection_finds_clause_heads_only() -> None:
    heads = _imperative_verbs("please summarize the forecast call and then draft a reply.")
    assert "summarize" in heads
    assert "draft" in heads
    assert "forecast" not in heads


def test_length_prior_only_decides_when_nothing_lexical_fires() -> None:
    # Short + a drafting verb must stay standard: length must not out-vote intent.
    short_with_verb = Classifier().heuristic(request("Explain our standard payment terms."))
    assert short_with_verb.complexity is Complexity.STANDARD
    # No verbs at all: the length prior is all there is, and it is weak.
    bare = Classifier().heuristic(
        request(
            "Northwind Hospitals Group, renewal Q4 2026, policy section 3.2, incumbent "
            "contract, procurement freeze until October, champion on leave, no quote yet."
        )
    )
    assert bare.confidence < ClassifierSettings().confidence_threshold


def test_empty_signal_defaults_to_standard_with_zero_confidence() -> None:
    verdict = Classifier()._score({})
    assert verdict == (Complexity.STANDARD, 0.0)


def test_estimate_tokens_is_four_characters_per_token() -> None:
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 1


def test_heuristic_is_pure() -> None:
    classifier = Classifier()
    req = request("Draft a renewal note for the champion.")
    assert classifier.heuristic(req) == classifier.heuristic(req)


# --------------------------------------------------------------------------- #
# Prompt hashing and the LRU cache
# --------------------------------------------------------------------------- #


def test_prompt_hash_ignores_max_tokens_but_not_the_workflow_hint() -> None:
    base = request("Extract the quote number.")
    louder = base.model_copy(update={"max_tokens": 4096})
    hinted = request("Extract the quote number.", workflow="discount_analyst")
    assert prompt_hash(base) == prompt_hash(louder)
    assert prompt_hash(base) != prompt_hash(hinted)


def test_lru_evicts_oldest_and_refreshes_on_read() -> None:
    cache = LRUCache(maxsize=2)
    cache.put("a", Complexity.TRIVIAL)
    cache.put("b", Complexity.STANDARD)
    assert cache.get("a") is Complexity.TRIVIAL  # refreshes "a", so "b" is oldest
    cache.put("c", Complexity.COMPLEX)
    assert len(cache) == 2
    assert "b" not in cache
    assert cache.get("b") is None
    assert cache.get("a") is Complexity.TRIVIAL


def test_lru_of_size_zero_is_a_disabled_cache() -> None:
    cache = LRUCache(maxsize=0)
    cache.put("a", Complexity.TRIVIAL)
    assert len(cache) == 0
    assert cache.get("a") is None


# --------------------------------------------------------------------------- #
# Stage 2 tie-break
# --------------------------------------------------------------------------- #

# Medium length, no verbs, no markers: the length prior is the only signal, and
# it is weak on purpose. This is what stage 2 exists for.
AMBIGUOUS = (
    "Northwind Hospitals Group, renewal Q4 2026, policy section 3.2, incumbent "
    "contract, procurement freeze until October, champion on leave, no quote yet, "
    "deal desk notified, security questionnaire outstanding."
)
OTHER_AMBIGUOUS = (
    "Acme Robotics, PO 4471, terms section 3.1, awaiting signature, partner tier "
    "silver, regional approval pending, evaluation hardware still on site, invoice "
    "on hold, renewal owner unassigned."
)


def test_fake_provider_satisfies_the_frozen_protocol() -> None:
    assert isinstance(FakeProvider(), Provider)


async def test_high_confidence_never_calls_the_tie_break() -> None:
    provider = FakeProvider("complex")
    classifier = Classifier(provider=provider, tiebreak_model="mock:echo")
    verdict = await classifier.classify(request("Extract the quote number. Return JSON only."))
    assert verdict.complexity is Complexity.TRIVIAL
    assert provider.call_count == 0
    assert classifier.stats.heuristic_resolved == 1


async def test_low_confidence_escalates_to_the_tie_break() -> None:
    provider = FakeProvider("complex")
    classifier = Classifier(provider=provider, tiebreak_model="mock:echo")
    verdict = await classifier.classify(request(AMBIGUOUS))
    assert provider.call_count == 1
    assert verdict.complexity is Complexity.COMPLEX
    assert verdict.source == "llm_tiebreak"
    assert classifier.stats.tiebreaks == 1
    # The tie-break runs on the cheap model it was handed, not the request's.
    _, model = provider.calls[0]
    assert model == "mock:echo"


async def test_repeated_prompt_hits_the_cache_with_zero_llm_calls() -> None:
    provider = FakeProvider("standard")
    classifier = Classifier(provider=provider, tiebreak_model="mock:echo")
    first = await classifier.classify(request(AMBIGUOUS))
    second = await classifier.classify(request(AMBIGUOUS))
    assert provider.call_count == 1
    assert second.source == "cache"
    assert second.complexity is first.complexity
    assert classifier.stats.cache_hits == 1


async def test_cache_eviction_forces_a_second_call() -> None:
    provider = FakeProvider("standard")
    classifier = Classifier(
        ClassifierSettings(cache_size=1), provider=provider, tiebreak_model="mock:echo"
    )
    await classifier.classify(request(AMBIGUOUS))
    await classifier.classify(request(OTHER_AMBIGUOUS))
    await classifier.classify(request(AMBIGUOUS))
    assert provider.call_count == 3


async def test_tie_break_failure_degrades_to_the_heuristic() -> None:
    provider = FakeProvider(fail=True)
    classifier = Classifier(provider=provider, tiebreak_model="mock:echo")
    verdict = await classifier.classify(request(AMBIGUOUS))
    assert verdict.source == "heuristic"
    assert classifier.stats.tiebreak_errors == 1


async def test_unparseable_tie_break_answer_degrades_to_the_heuristic() -> None:
    provider = FakeProvider("I am not going to answer that")
    classifier = Classifier(provider=provider, tiebreak_model="mock:echo")
    verdict = await classifier.classify(request(AMBIGUOUS))
    assert verdict.source == "heuristic"
    assert classifier.stats.tiebreak_errors == 1


async def test_tie_break_reads_the_first_tier_word_in_the_answer() -> None:
    provider = FakeProvider("standard, not complex")
    classifier = Classifier(provider=provider, tiebreak_model="mock:echo")
    verdict = await classifier.classify(request(AMBIGUOUS))
    assert verdict.complexity is Complexity.STANDARD


async def test_tie_break_disabled_by_settings_makes_no_calls() -> None:
    provider = FakeProvider("complex")
    classifier = Classifier(
        ClassifierSettings(tiebreak_enabled=False), provider=provider, tiebreak_model="mock:echo"
    )
    verdict = await classifier.classify(request(AMBIGUOUS))
    assert provider.call_count == 0
    assert verdict.source == "heuristic"


async def test_threshold_controls_escalation() -> None:
    """The same prompt escalates or not purely on where the threshold sits."""
    borderline = request("Draft a follow-up email to the champion about pricing.")

    lax = Classifier(
        ClassifierSettings(confidence_threshold=0.5),
        provider=(lax_provider := FakeProvider("complex")),
        tiebreak_model="mock:echo",
    )
    assert (await lax.classify(borderline)).source == "heuristic"
    assert lax_provider.call_count == 0

    strict = Classifier(
        ClassifierSettings(confidence_threshold=0.99),
        provider=(strict_provider := FakeProvider("complex")),
        tiebreak_model="mock:echo",
    )
    assert (await strict.classify(borderline)).source == "llm_tiebreak"
    assert strict_provider.call_count == 1


async def test_no_provider_means_no_escalation() -> None:
    classifier = Classifier()
    verdict = await classifier.classify(request(AMBIGUOUS))
    assert verdict.source == "heuristic"
    assert classifier.stats.tiebreaks == 0


def test_stats_report_the_heuristic_rate() -> None:
    classifier = Classifier()
    assert classifier.stats.heuristic_rate == 0.0
    classifier.stats.classified = 4
    classifier.stats.heuristic_resolved = 3
    assert classifier.stats.heuristic_rate == pytest.approx(0.75)
