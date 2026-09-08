"""The offline judge: a `Provider` that grades rubrics without a network call.

CI must run every suite at zero API spend (epic AC-5) and the gate must be
reproducible, which rules out a real judge model on a pull request. The obvious
shortcut — special-case the judge away in tests — would leave the prompt
builder, the JSON parser and the retry path untested on the only run that
gates a merge.

So the stand-in is a provider, not a mock. It reads the same delimited judge
prompt a real model would receive, evaluates the rubric criteria mechanically,
and emits the same JSON contract. Every layer above it is exercised unchanged.

What it is not: a language model. It cannot judge anything a rubric criterion
cannot express. Point `judge_model` at a real model id for a semantic verdict;
the offline judge is what makes the *regression* signal free.
"""

import json
import time

from conduit.contracts import CompletionRequest, CompletionResponse, Complexity, Usage

from .judge import extract_sections
from .rubric import RubricError, evaluate_rubric, parse_rubric

__all__ = ["OFFLINE_JUDGE_MODEL", "OfflineJudgeProvider"]

OFFLINE_JUDGE_MODEL = "offline-judge:rubric"


class OfflineJudgeProvider:
    """Implements `conduit.contracts.Provider`. Deterministic, offline, free."""

    name = "offline-judge"

    def __init__(self, *, pass_threshold: float = 0.7) -> None:
        self.pass_threshold = pass_threshold
        self.call_count = 0

    def supports(self, model: str) -> bool:
        return model == OFFLINE_JUDGE_MODEL

    async def health(self) -> bool:
        return True

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        started = time.perf_counter()
        self.call_count += 1
        prompt = next(
            (
                message.content
                for message in reversed(req.messages)
                if "[CANDIDATE]" in message.content
            ),
            "",
        )
        text = json.dumps(self._grade(prompt), separators=(",", ": "))
        return CompletionResponse(
            text=text,
            model=model,
            provider=self.name,
            usage=Usage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0),
            latency_ms=int((time.perf_counter() - started) * 1000),
            routed_tier=Complexity.TRIVIAL,
        )

    def _grade(self, prompt: str) -> dict[str, object]:
        sections = extract_sections(prompt)
        source = sections.get("RUBRIC_SOURCE", "").strip()
        candidate = sections.get("CANDIDATE", "")
        reference = sections.get("REFERENCE", "")
        if not source:
            # A prompt with no rubric is a bug in the caller, and reporting it
            # as a parse failure is what surfaces it: the runner records an
            # error rather than a zero.
            return {"error": "no rubric in prompt"}
        try:
            rubric = parse_rubric(source)
        except RubricError as exc:
            return {"error": str(exc)}
        results = evaluate_rubric(rubric, candidate, reference)
        satisfied = [result for result in results if result.satisfied]
        score = len(satisfied) / len(results)
        failed = [result.criterion.render() for result in results if not result.satisfied]
        reasoning = f"{len(satisfied)}/{len(results)} criteria satisfied" + (
            f"; unmet: {', '.join(failed)}" if failed else ""
        )
        return {
            "score": round(score, 4),
            "passed": score >= self.pass_threshold,
            "reasoning": reasoning,
        }
