"""Rubric parsing and criterion evaluation — the unit the score is built from."""

import pytest

from conduit.eval.rubric import (
    RubricError,
    evaluate_rubric,
    parse_rubric,
    token_f1,
)


def test_parse_splits_criteria_and_tolerates_whitespace() -> None:
    rubric = parse_rubric("  contains:rvp ;  min_words:15 ;; no_placeholder  ")
    assert [c.name for c in rubric.criteria] == ["contains", "min_words", "no_placeholder"]
    assert rubric.criteria[0].argument == "rvp"
    assert rubric.criteria[2].argument == ""
    assert rubric.render() == "contains:rvp; min_words:15; no_placeholder"


def test_an_unknown_criterion_is_an_error_not_a_skip() -> None:
    """A silently dropped criterion inflates every score computed after it."""
    with pytest.raises(RubricError, match="unknown rubric criterion 'vibes'"):
        parse_rubric("contains:rvp; vibes:good")


@pytest.mark.parametrize(
    "source, message",
    [
        ("", "empty"),
        ("   ", "empty"),
        (";;;", "no criteria"),
        ("contains", "requires an argument"),
        ("no_placeholder:x", "takes no argument"),
        ("min_words:many", "whole number"),
        ("similar_to_reference:high", "needs a number"),
        ("similar_to_reference:1.5", "within 0.0-1.0"),
        ("matches:[unclosed", "invalid regex"),
    ],
)
def test_malformed_rubrics_are_rejected_with_a_reason(source: str, message: str) -> None:
    with pytest.raises(RubricError, match=message):
        parse_rubric(source)


def test_contains_and_mentions_any_are_case_insensitive() -> None:
    rubric = parse_rubric("contains:RVP; mentions_any:20%|twenty percent")
    results = evaluate_rubric(rubric, "Needs rvp sign-off above twenty percent.")
    assert [r.satisfied for r in results] == [True, True]


def test_word_count_criteria_bound_the_answer_from_both_sides() -> None:
    rubric = parse_rubric("min_words:3; max_words:5")
    assert [r.satisfied for r in evaluate_rubric(rubric, "one two three four")] == [True, True]
    assert [r.satisfied for r in evaluate_rubric(rubric, "too short")] == [False, True]
    assert [r.satisfied for r in evaluate_rubric(rubric, "a b c d e f")] == [True, False]


def test_no_placeholder_catches_unrehydrated_redaction_output() -> None:
    """Epic AC-8: the caller must never see a placeholder."""
    rubric = parse_rubric("no_placeholder")
    assert evaluate_rubric(rubric, "Send it to Jane Doe.")[0].satisfied
    leaked = evaluate_rubric(rubric, "Send it to <PERSON_1> at <EMAIL_ADDRESS_2>.")[0]
    assert not leaked.satisfied
    assert "PERSON_1" in leaked.detail


def test_similar_to_reference_uses_the_reference_not_the_input() -> None:
    rubric = parse_rubric("similar_to_reference:0.5")
    close = evaluate_rubric(rubric, "the credit is prorated", "the credit is prorated")
    far = evaluate_rubric(rubric, "unrelated words entirely", "the credit is prorated")
    assert close[0].satisfied and not far[0].satisfied


def test_token_f1_is_symmetric_bounded_and_zero_on_empty() -> None:
    assert token_f1("a b c", "a b c") == pytest.approx(1.0)
    assert token_f1("a b c", "c b a") == pytest.approx(1.0)
    assert token_f1("a b", "c d") == 0.0
    assert token_f1("", "a") == 0.0
    assert 0.0 < token_f1("a b c d", "a b") < 1.0


def test_evaluation_preserves_rubric_order() -> None:
    rubric = parse_rubric("max_words:2; contains:zebra; min_words:1")
    results = evaluate_rubric(rubric, "zebra")
    assert [r.criterion.name for r in results] == ["max_words", "contains", "min_words"]
