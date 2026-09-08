"""LLM-as-a-Judge: rubric in, `JudgeScore` out, deterministically parsed.

Three properties matter more than the prompt wording.

*Parsing is strict.* The judge is asked for one JSON object and nothing else.
The parser takes the first balanced object in the response and validates every
field. A response it cannot read is an error, never a zero — a silent zero is
indistinguishable from a genuinely bad answer and would quietly move a gate.

*Malformed responses get exactly one retry.* The retry restates the format
demand rather than the question, so a model that drifted into prose gets a
second chance without the case being re-asked in a way that changes it.

*The prompt is machine-readable in both directions.* Sections are delimited by
`[NAME] ... [/NAME]`, which is what lets `OfflineJudgeProvider` implement the
same `Provider` protocol as a real judge model and keeps the CI path identical
to the production one instead of a mock that skips the parser.
"""

import json
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from conduit.contracts import CompletionRequest, JudgeScore, Message, Provider

from .dataset import EvalCase
from .rubric import Rubric, parse_rubric

__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "Judge",
    "JudgeError",
    "JudgeParseError",
    "JudgeResult",
    "build_judge_prompt",
    "extract_sections",
    "parse_judge_response",
]

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. You are given a rubric, the input a model was "
    "given, a reference answer, and the candidate answer to grade. Check the "
    "candidate against every numbered rubric criterion. Reply with exactly one "
    "JSON object and no other text:\n"
    '{"score": <fraction of criteria satisfied, 0.0-1.0>, '
    '"passed": <true|false>, "reasoning": "<one sentence>"}'
)

_RETRY_NUDGE = (
    "Your previous reply was not valid JSON. Reply with exactly one JSON object "
    'of the form {"score": 0.0, "passed": false, "reasoning": "..."} and nothing else.'
)

_SECTION = re.compile(r"\[([A-Z_]+)\]\n(.*?)\n\[/\1\]", re.DOTALL)


class JudgeError(Exception):
    """The judge could not produce a usable score for a case."""


class JudgeParseError(JudgeError):
    """The judge replied with something that is not a scored verdict."""


class JudgeResult(BaseModel):
    """A `JudgeScore` plus what it cost to obtain it.

    The score itself is the contract type; the surrounding fields exist because
    epic AC-6 wants eval spend to be as observable as request spend.
    """

    score: JudgeScore
    attempts: int = Field(ge=1)
    cost_usd: float = 0.0
    latency_ms: int = 0
    model: str = ""


def build_judge_prompt(case: EvalCase, candidate: str, rubric: Rubric) -> str:
    """Render one grading task. Sections are delimited so they parse back."""
    criteria = "\n".join(
        f"{index}. {criterion.render()}" for index, criterion in enumerate(rubric.criteria, start=1)
    )
    return (
        f"[RUBRIC]\n{criteria}\n[/RUBRIC]\n"
        f"[RUBRIC_SOURCE]\n{rubric.source}\n[/RUBRIC_SOURCE]\n"
        f"[INPUT]\n{case.input}\n[/INPUT]\n"
        f"[REFERENCE]\n{case.expected}\n[/REFERENCE]\n"
        f"[CANDIDATE]\n{candidate}\n[/CANDIDATE]"
    )


def extract_sections(prompt: str) -> dict[str, str]:
    """Pull `[NAME] ... [/NAME]` blocks back out of a judge prompt."""
    return dict(_SECTION.findall(prompt))


def _first_json_object(text: str) -> str:
    """The first balanced `{...}` run, ignoring braces inside strings."""
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise JudgeParseError("no JSON object found in judge response")


def parse_judge_response(text: str, rubric: str, *, pass_threshold: float = 0.7) -> JudgeScore:
    """Parse a judge reply into a `JudgeScore`. Raises `JudgeParseError`.

    `passed` is taken from the judge when it supplies one and derived from the
    threshold otherwise, so a judge that returns only a score is still usable.
    """
    blob = _first_json_object(text)
    try:
        payload: Any = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"judge response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeParseError(f"judge response must be a JSON object, got {type(payload).__name__}")
    if "score" not in payload:
        raise JudgeParseError(f"judge response has no 'score' field: {sorted(payload)}")
    raw_score = payload["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
        raise JudgeParseError(f"judge 'score' must be a number, got {raw_score!r}")
    score = float(raw_score)
    if not 0.0 <= score <= 1.0:
        raise JudgeParseError(f"judge 'score' must be within 0.0-1.0, got {score}")
    raw_passed = payload.get("passed")
    if raw_passed is None:
        passed = score >= pass_threshold
    elif isinstance(raw_passed, bool):
        passed = raw_passed
    else:
        raise JudgeParseError(f"judge 'passed' must be a boolean, got {raw_passed!r}")
    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise JudgeParseError(f"judge 'reasoning' must be a string, got {reasoning!r}")
    return JudgeScore(rubric=rubric, score=score, reasoning=reasoning.strip(), passed=passed)


class Judge:
    """Scores candidate answers against a rubric using a pinned judge model.

    The judge model is passed in, never inferred from the request under test.
    `EvalSettings` is what guarantees the two differ; this class simply uses
    what it is given.
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        retries: int = 1,
        max_tokens: int = 512,
    ) -> None:
        self.provider = provider
        self.model = model
        self.retries = retries
        self.max_tokens = max_tokens

    def _request(self, prompt: str, *, nudge: bool) -> CompletionRequest:
        messages = [
            Message(role="system", content=JUDGE_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]
        if nudge:
            messages.append(Message(role="user", content=_RETRY_NUDGE))
        return CompletionRequest(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

    async def score(self, case: EvalCase, candidate: str, *, pass_threshold: float) -> JudgeResult:
        """Grade one candidate answer. Raises `JudgeError` after the last retry."""
        rubric = parse_rubric(case.rubric)
        prompt = build_judge_prompt(case, candidate, rubric)
        cost = 0.0
        started = time.perf_counter()
        last: Exception | None = None
        for attempt in range(1, self.retries + 2):
            request = self._request(prompt, nudge=attempt > 1)
            try:
                response = await self.provider.complete(request, self.model)
            except Exception as exc:  # provider failures are judge errors too
                last = exc
                continue
            cost += response.usage.cost_usd
            try:
                verdict = parse_judge_response(
                    response.text, case.rubric, pass_threshold=pass_threshold
                )
            except JudgeParseError as exc:
                last = exc
                continue
            return JudgeResult(
                score=verdict,
                attempts=attempt,
                cost_usd=cost,
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=response.model,
            )
        raise JudgeError(
            f"judge {self.model!r} failed on case {case.id!r} after "
            f"{self.retries + 1} attempt(s): {last}"
        ) from last
