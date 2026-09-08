"""Judge parsing, the single retry, and the rule that errors are never zeros."""

import pytest

from conduit.contracts import CompletionRequest, CompletionResponse, Complexity, Message, Usage
from conduit.eval.dataset import EvalCase
from conduit.eval.judge import (
    Judge,
    JudgeError,
    JudgeParseError,
    build_judge_prompt,
    extract_sections,
    parse_judge_response,
)
from conduit.eval.offline import OFFLINE_JUDGE_MODEL, OfflineJudgeProvider
from conduit.eval.rubric import parse_rubric

CASE = EvalCase(
    id="j-1",
    input="Does 22% need RVP approval?",
    expected="Yes, above the 20% line it needs RVP sign-off.",
    rubric="contains:rvp; min_words:5",
)


class ScriptedProvider:
    """Returns canned judge replies in order, so retry behaviour is observable."""

    name = "scripted"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[CompletionRequest] = []

    def supports(self, model: str) -> bool:
        return True

    async def health(self) -> bool:
        return True

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        self.calls.append(req)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return CompletionResponse(
            text=reply,
            model=model,
            provider=self.name,
            usage=Usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.001),
            latency_ms=1,
            routed_tier=Complexity.TRIVIAL,
        )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_accepts_a_bare_object_and_derives_passed_from_the_threshold() -> None:
    score = parse_judge_response('{"score": 0.8, "reasoning": "good"}', "r", pass_threshold=0.7)
    assert score.score == 0.8
    assert score.passed is True
    assert score.reasoning == "good"
    assert score.rubric == "r"


def test_parse_finds_the_object_inside_surrounding_prose() -> None:
    text = (
        'Sure! Here is my verdict:\n```json\n{"score": 0.5, "passed": false}\n```\nHope that helps.'
    )
    assert parse_judge_response(text, "r").score == 0.5


def test_parse_ignores_braces_inside_strings() -> None:
    text = '{"score": 1.0, "reasoning": "the answer used {placeholders}"}'
    assert parse_judge_response(text, "r").reasoning == "the answer used {placeholders}"


def test_an_explicit_passed_flag_overrides_the_threshold() -> None:
    score = parse_judge_response('{"score": 0.9, "passed": false}', "r", pass_threshold=0.5)
    assert score.score == 0.9 and score.passed is False


@pytest.mark.parametrize(
    "text, message",
    [
        ("no json here", "no JSON object"),
        ("{not json}", "not valid JSON"),
        ("[1, 2]", "no JSON object"),
        ('{"reasoning": "forgot the score"}', "no 'score' field"),
        ('{"score": "high"}', "must be a number"),
        ('{"score": true}', "must be a number"),
        ('{"score": 1.4}', "within 0.0-1.0"),
        ('{"score": -0.1}', "within 0.0-1.0"),
        ('{"score": 0.5, "passed": "yes"}', "must be a boolean"),
        ('{"score": 0.5, "reasoning": 7}', "must be a string"),
    ],
)
def test_malformed_judge_output_raises_rather_than_scoring_zero(text: str, message: str) -> None:
    """Epic AC-4: a zero and an unreadable reply must not look the same."""
    with pytest.raises(JudgeParseError, match=message):
        parse_judge_response(text, "r")


# --------------------------------------------------------------------------- #
# Prompt round trip
# --------------------------------------------------------------------------- #


def test_the_judge_prompt_parses_back_into_its_sections() -> None:
    prompt = build_judge_prompt(CASE, "candidate answer", parse_rubric(CASE.rubric))
    sections = extract_sections(prompt)
    assert sections["RUBRIC_SOURCE"] == CASE.rubric
    assert sections["INPUT"] == CASE.input
    assert sections["REFERENCE"] == CASE.expected
    assert sections["CANDIDATE"] == "candidate answer"
    assert "1. contains:rvp" in sections["RUBRIC"]


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


async def test_a_malformed_reply_is_retried_exactly_once_then_succeeds() -> None:
    provider = ScriptedProvider("I think it is pretty good", '{"score": 1.0, "passed": true}')
    result = await Judge(provider, "judge-model", retries=1).score(
        CASE, "answer", pass_threshold=0.7
    )
    assert result.attempts == 2
    assert result.score.score == 1.0
    assert len(provider.calls) == 2
    # The retry restates the format demand rather than re-asking the question.
    assert "valid JSON" in provider.calls[1].messages[-1].content


async def test_persistent_malformed_output_raises_and_is_never_scored_zero() -> None:
    provider = ScriptedProvider("still prose", "more prose")
    with pytest.raises(JudgeError, match="after 2 attempt"):
        await Judge(provider, "judge-model", retries=1).score(CASE, "answer", pass_threshold=0.7)
    assert len(provider.calls) == 2


async def test_retries_zero_means_a_single_attempt() -> None:
    provider = ScriptedProvider("prose")
    with pytest.raises(JudgeError, match="after 1 attempt"):
        await Judge(provider, "judge-model", retries=0).score(CASE, "answer", pass_threshold=0.7)
    assert len(provider.calls) == 1


async def test_judge_cost_accumulates_across_attempts() -> None:
    """AC-6: eval spend is observable, including the spend a retry caused."""
    provider = ScriptedProvider("prose", '{"score": 0.5}')
    result = await Judge(provider, "judge-model", retries=1).score(
        CASE, "answer", pass_threshold=0.7
    )
    assert result.cost_usd == pytest.approx(0.002)


# --------------------------------------------------------------------------- #
# Offline judge
# --------------------------------------------------------------------------- #


async def test_the_offline_judge_scores_the_fraction_of_criteria_satisfied() -> None:
    judge = Judge(OfflineJudgeProvider(), OFFLINE_JUDGE_MODEL)
    half = await judge.score(CASE, "rvp", pass_threshold=0.7)
    assert half.score.score == pytest.approx(0.5)
    assert half.score.passed is False
    assert "min_words:5" in half.score.reasoning
    full = await judge.score(CASE, "yes it needs rvp sign-off above the line", pass_threshold=0.7)
    assert full.score.score == pytest.approx(1.0)
    assert full.score.passed is True


async def test_the_offline_judge_is_deterministic_and_free() -> None:
    provider = OfflineJudgeProvider()
    judge = Judge(provider, OFFLINE_JUDGE_MODEL)
    first = await judge.score(CASE, "needs rvp sign-off above the line", pass_threshold=0.7)
    second = await judge.score(CASE, "needs rvp sign-off above the line", pass_threshold=0.7)
    assert first.score.model_dump() == second.score.model_dump()
    assert first.cost_usd == 0.0


async def test_a_prompt_with_no_rubric_becomes_an_error_rather_than_a_zero() -> None:
    """The offline judge refuses to invent a score it has no criteria for."""
    provider = OfflineJudgeProvider()
    response = await provider.complete(
        CompletionRequest(messages=[Message(role="user", content="[CANDIDATE]\nx\n[/CANDIDATE]")]),
        OFFLINE_JUDGE_MODEL,
    )
    assert "no rubric" in response.text
    with pytest.raises(JudgeParseError, match="no 'score' field"):
        parse_judge_response(response.text, "r")


def test_the_offline_judge_only_claims_to_serve_its_own_model_id() -> None:
    provider = OfflineJudgeProvider()
    assert provider.supports(OFFLINE_JUDGE_MODEL)
    assert not provider.supports("claude-opus-5")
