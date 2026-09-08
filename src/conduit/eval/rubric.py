"""The rubric mini-language shared by the LLM judge and the offline judge.

A rubric is a semicolon-separated list of `name:argument` criteria::

    contains:rvp; mentions_any:20%|twenty percent; min_words:15; no_placeholder

Making the rubric parseable rather than free prose buys three things. The judge
prompt renders it as numbered criteria, so an LLM judge is asked a bounded
question instead of "is this good". The offline judge used in CI evaluates the
same criteria mechanically, so the gate is reproducible and free. And a
criterion is the unit of score: `score = satisfied / total`, which is a number
that survives a model swap, unlike a holistic 1-5 opinion.

Unknown criterion names are an error, not a skip. A typo that silently drops a
criterion inflates every score that follows it.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel

__all__ = [
    "CRITERIA",
    "Criterion",
    "CriterionResult",
    "Rubric",
    "RubricError",
    "evaluate_rubric",
    "parse_rubric",
    "token_f1",
]


class RubricError(ValueError):
    """Raised when a rubric string cannot be parsed."""


#: Criterion name -> whether it takes an argument.
CRITERIA: dict[str, bool] = {
    "contains": True,
    "mentions_any": True,
    "matches": True,
    "min_words": True,
    "max_words": True,
    "similar_to_reference": True,
    "no_placeholder": False,
}

_PLACEHOLDER = re.compile(r"<[A-Z][A-Z_]*_\d+>")
_WORD = re.compile(r"[\w$%\-,.]+")


class Criterion(BaseModel):
    """One checkable clause of a rubric."""

    name: str
    argument: str = ""

    def render(self) -> str:
        return f"{self.name}:{self.argument}" if self.argument else self.name


class Rubric(BaseModel):
    """An ordered, non-empty list of criteria."""

    source: str
    criteria: list[Criterion]

    def render(self) -> str:
        return "; ".join(criterion.render() for criterion in self.criteria)


class CriterionResult(BaseModel):
    criterion: Criterion
    satisfied: bool
    detail: str = ""


def parse_rubric(source: str) -> Rubric:
    """Parse a rubric string. Raises `RubricError` on anything unrecognised."""
    if not source or not source.strip():
        raise RubricError("rubric is empty")
    criteria: list[Criterion] = []
    for clause in source.split(";"):
        text = clause.strip()
        if not text:
            continue
        name, separator, argument = text.partition(":")
        name = name.strip()
        argument = argument.strip()
        if name not in CRITERIA:
            known = ", ".join(sorted(CRITERIA))
            raise RubricError(f"unknown rubric criterion {name!r}; known criteria: {known}")
        takes_argument = CRITERIA[name]
        if takes_argument and not argument:
            raise RubricError(f"criterion {name!r} requires an argument, e.g. {name}:<value>")
        if not takes_argument and separator:
            raise RubricError(f"criterion {name!r} takes no argument, got {argument!r}")
        if name in {"min_words", "max_words"} and not argument.isdigit():
            raise RubricError(f"criterion {name!r} needs a whole number, got {argument!r}")
        if name == "similar_to_reference":
            try:
                threshold = float(argument)
            except ValueError as exc:
                raise RubricError(
                    f"criterion 'similar_to_reference' needs a number, got {argument!r}"
                ) from exc
            if not 0.0 <= threshold <= 1.0:
                raise RubricError(
                    f"criterion 'similar_to_reference' must be within 0.0-1.0, got {threshold}"
                )
        if name == "matches":
            try:
                re.compile(argument)
            except re.error as exc:
                raise RubricError(f"criterion 'matches' has an invalid regex: {exc}") from exc
        criteria.append(Criterion(name=name, argument=argument))
    if not criteria:
        raise RubricError("rubric contains no criteria")
    return Rubric(source=source, criteria=criteria)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def token_f1(candidate: str, reference: str) -> float:
    """Bag-of-words F1 between two texts, 0.0-1.0.

    Crude on purpose: it is a stable similarity floor, not a semantic score. The
    LLM judge is where semantics live; this is what keeps CI free and
    deterministic.
    """
    left, right = _words(candidate), _words(reference)
    if not left or not right:
        return 0.0
    overlap = 0
    remaining = list(right)
    for token in left:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall)


def _check(criterion: Criterion, candidate: str, reference: str) -> CriterionResult:
    lowered = candidate.lower()
    name, argument = criterion.name, criterion.argument
    if name == "contains":
        ok = argument.lower() in lowered
        return CriterionResult(criterion=criterion, satisfied=ok, detail=f"contains {argument!r}")
    if name == "mentions_any":
        options = [option.strip().lower() for option in argument.split("|") if option.strip()]
        hit = next((option for option in options if option in lowered), None)
        return CriterionResult(
            criterion=criterion,
            satisfied=hit is not None,
            detail=f"matched {hit!r}" if hit else f"none of {options}",
        )
    if name == "matches":
        ok = re.search(argument, candidate) is not None
        return CriterionResult(criterion=criterion, satisfied=ok, detail=f"regex {argument!r}")
    if name == "min_words":
        count = len(_words(candidate))
        return CriterionResult(
            criterion=criterion, satisfied=count >= int(argument), detail=f"{count} words"
        )
    if name == "max_words":
        count = len(_words(candidate))
        return CriterionResult(
            criterion=criterion, satisfied=count <= int(argument), detail=f"{count} words"
        )
    if name == "similar_to_reference":
        score = token_f1(candidate, reference)
        return CriterionResult(
            criterion=criterion, satisfied=score >= float(argument), detail=f"f1={score:.2f}"
        )
    # `no_placeholder` — the only zero-argument criterion, so it is the default.
    leftovers = _PLACEHOLDER.findall(candidate)
    return CriterionResult(
        criterion=criterion,
        satisfied=not leftovers,
        detail=f"leftover placeholders: {leftovers}" if leftovers else "no placeholders",
    )


def evaluate_rubric(
    rubric: Rubric, candidate: str, reference: str = ""
) -> Sequence[CriterionResult]:
    """Check every criterion against a candidate answer, in rubric order."""
    return [_check(criterion, candidate, reference) for criterion in rubric.criteria]
