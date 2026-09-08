"""Golden datasets: the JSONL on disk and the record it parses into.

The epic fixes the row shape as `{id, input, expected, rubric, tier}`. `expected`
is deliberately a plain string rather than a per-suite enum: routing stores a
tier, the guard suite stores a decision, and the answer-quality suite stores a
whole reference answer. The suite's scorer is what gives the field meaning, so
the loader stays one function instead of three.

`metadata` carries provenance. Every row says which issue authored it, because
a dataset whose author also tuned the system against it is not a measurement —
see `docs/EVAL.md`.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from conduit.contracts import CompletionRequest, Message

__all__ = [
    "DatasetError",
    "EvalCase",
    "default_golden_dir",
    "load_cases",
]


class DatasetError(Exception):
    """Raised when a golden dataset is missing, unparseable, or malformed."""


class EvalCase(BaseModel):
    """One labelled example. Schema matches the epic's golden format."""

    id: str
    input: str
    expected: str
    rubric: str = ""
    tier: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("id", "input")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @property
    def slice(self) -> str:
        """The sub-population this case belongs to, for per-slice reporting."""
        return self.metadata.get("slice", "default")

    def to_request(self) -> CompletionRequest:
        return CompletionRequest(
            messages=[Message(role="user", content=self.input)],
            metadata={k: v for k, v in self.metadata.items() if k != "slice"},
        )


def default_golden_dir() -> Path:
    """`evals/golden/` at the repo root."""
    return Path(__file__).resolve().parents[3] / "evals" / "golden"


def load_cases(path: Path) -> list[EvalCase]:
    """Read a golden JSONL, one case per non-empty line.

    Duplicate ids are rejected rather than deduplicated: a repeated id silently
    reweights the suite, which would move a pass rate without moving quality.
    """
    target = Path(path)
    if not target.is_file():
        raise DatasetError(f"missing golden dataset: {target}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{target}:{number}: invalid JSON: {exc}") from exc
        try:
            case = EvalCase.model_validate(payload)
        except ValueError as exc:
            raise DatasetError(f"{target}:{number}: invalid case: {exc}") from exc
        if case.id in seen:
            raise DatasetError(f"{target}:{number}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError(f"{target} contains no cases")
    return cases
