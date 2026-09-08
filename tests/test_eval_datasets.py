"""The golden datasets themselves: shape, balance, and provenance.

An eval that only confirms its author is worthless, so provenance is asserted
rather than documented. Every routing case records who labelled it, and the
suite carries a slice that was written by someone other than the component's
author. See docs/EVAL.md.
"""

from collections import Counter
from pathlib import Path

import pytest

from conduit.eval.dataset import DatasetError, EvalCase, load_cases
from conduit.eval.rubric import parse_rubric
from conduit.eval.settings import load_eval

SETTINGS = load_eval()


def cases_for(name: str) -> list[EvalCase]:
    spec = SETTINGS.suite(name)
    return load_cases(SETTINGS.resolve_path(spec))


@pytest.mark.parametrize("name", sorted(SETTINGS.suites))
def test_every_configured_suite_loads(name: str) -> None:
    assert cases_for(name)


def test_routing_labels_were_re_authored_under_this_issue() -> None:
    """The integrity claim, asserted: no routing label is self-graded."""
    cases = cases_for("routing")
    assert all(case.metadata.get("labeler") == "issue-#8" for case in cases)
    origins = Counter(case.metadata.get("origin") for case in cases)
    assert origins["inherited-#4"] == 60
    assert origins["independent-#8"] == 45


def test_routing_covers_every_tier_without_one_dominating() -> None:
    cases = cases_for("routing")
    counts = Counter(case.expected for case in cases)
    assert set(counts) == {"trivial", "standard", "complex"}
    for share in (count / len(cases) for count in counts.values()):
        assert 0.2 <= share <= 0.5


def test_the_guard_corpus_measures_false_positives_not_only_recall() -> None:
    """A block-everything guard must be able to fail this suite."""
    cases = cases_for("guards")
    counts = Counter(case.expected for case in cases)
    assert counts["block"] == 30
    assert counts["allow"] == 20
    assert counts["redact"] == 10
    families = {case.metadata["family"] for case in cases if case.expected == "block"}
    assert len(families) >= 8


def test_every_quality_rubric_parses() -> None:
    """A typo'd criterion would silently inflate a score, so it is caught here."""
    for case in cases_for("l2c_quality"):
        assert parse_rubric(case.rubric).criteria


def test_case_ids_are_unique_within_every_suite() -> None:
    for name in SETTINGS.suites:
        cases = cases_for(name)
        assert len({case.id for case in cases}) == len(cases)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def test_a_duplicate_id_is_rejected_rather_than_deduplicated(tmp_path: Path) -> None:
    """A repeated id reweights the suite, moving a rate without moving quality."""
    path = tmp_path / "dupes.jsonl"
    path.write_text(
        '{"id": "a", "input": "x", "expected": "y"}\n{"id": "a", "input": "z", "expected": "y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate case id 'a'"):
        load_cases(path)


def test_a_malformed_line_names_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a", "input": "x", "expected": "y"}\nnot json\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"bad\.jsonl:2: invalid JSON"):
        load_cases(path)


def test_a_blank_input_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text('{"id": "a", "input": "   ", "expected": "y"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid case"):
        load_cases(path)


def test_an_empty_file_is_an_error_not_an_empty_pass(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="contains no cases"):
        load_cases(path)


def test_a_missing_file_is_reported_by_path(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="missing golden dataset"):
        load_cases(tmp_path / "nope.jsonl")


def test_the_slice_is_not_smuggled_into_request_metadata() -> None:
    case = EvalCase(
        id="a", input="x", expected="y", metadata={"slice": "hard", "origin": "independent-#8"}
    )
    assert case.slice == "hard"
    assert case.to_request().metadata == {"origin": "independent-#8"}
