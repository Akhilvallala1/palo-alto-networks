"""Loaders for the guard fixtures in `tests/fixtures/`.

Kept out of `conftest.py` on purpose: the corpora are shared by the unit tests
and the eval suite, but they are guard-specific and should not become ambient
fixtures for every test in the repo.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
PII_CORPUS = FIXTURES / "pii_corpus.jsonl"
INJECTION_CORPUS = FIXTURES / "injection_corpus.jsonl"


@dataclass(frozen=True)
class PIICase:
    id: str
    text: str
    entities: tuple[tuple[str, str], ...]  # (entity_type, exact original substring)

    @property
    def is_negative(self) -> bool:
        return not self.entities


@dataclass(frozen=True)
class InjectionCase:
    id: str
    text: str
    label: str  # "attack" | "benign"
    family: str

    @property
    def is_attack(self) -> bool:
        return self.label == "attack"


def _read(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@lru_cache(maxsize=1)
def pii_cases() -> tuple[PIICase, ...]:
    cases: list[PIICase] = []
    for row in _read(PII_CORPUS):
        entities = row["entities"]
        assert isinstance(entities, list)
        cases.append(
            PIICase(
                id=str(row["id"]),
                text=str(row["text"]),
                entities=tuple((str(e["type"]), str(e["value"])) for e in entities),
            )
        )
    return tuple(cases)


@lru_cache(maxsize=1)
def injection_cases() -> tuple[InjectionCase, ...]:
    return tuple(
        InjectionCase(
            id=str(row["id"]),
            text=str(row["text"]),
            label=str(row["label"]),
            family=str(row["family"]),
        )
        for row in _read(INJECTION_CORPUS)
    )


def attacks() -> tuple[InjectionCase, ...]:
    return tuple(case for case in injection_cases() if case.is_attack)


def benign() -> tuple[InjectionCase, ...]:
    return tuple(case for case in injection_cases() if not case.is_attack)
