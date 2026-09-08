"""Unit tests for the append-only JSONL log and its SQLite index."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from conduit.contracts import Complexity
from conduit.telemetry.store import TelemetryQuery, read_jsonl
from telemetry_fixtures import EPOCH, make_record, open_store, seed


def test_a_write_appends_one_line_and_one_indexed_row(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.write(make_record(index=1))

    lines = store.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace_id"] == "trace-0001"
    assert len(store.records()) == 1
    store.close()


def test_rows_round_trip_through_sqlite_with_lists_and_enums_intact(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    original = make_record(index=2, tier=Complexity.COMPLEX)
    original.guard_categories = ["pii:email"]
    original.fallback_from = ["claude-opus-5"]
    store.write(original)

    (row,) = store.records()
    assert row == original
    store.close()


def test_the_jsonl_log_replays_into_the_same_records(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    records = [make_record(index=i) for i in range(3)]
    seed(store, records)

    assert list(read_jsonl(store.jsonl_path)) == records
    assert list(read_jsonl(tmp_path / "nothing-here.jsonl")) == []
    store.close()


def test_the_index_is_derived_and_can_be_rebuilt_from_the_log(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    seed(store, [make_record(index=i) for i in range(4)])

    assert store.rebuild_from_jsonl() == 4
    assert len(store.records()) == 4
    store.close()


def test_a_replayed_row_is_not_double_counted(tmp_path: Path) -> None:
    """`record_id` is the primary key, so re-indexing the log cannot inflate cost."""
    store = open_store(tmp_path)
    record = make_record(index=5)
    store.write(record)
    store.write(record)

    assert len(store.records()) == 1
    store.close()


def test_queries_filter_by_time_window(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    seed(
        store,
        [
            make_record(index=0, minutes_ago=0),
            make_record(index=1, minutes_ago=30),
            make_record(index=2, minutes_ago=90),
        ],
    )

    recent = store.records(TelemetryQuery(since=EPOCH - timedelta(minutes=60)))
    assert {row.record_id for row in recent} == {"rec-0000", "rec-0001"}
    store.close()


def test_queries_filter_by_equality_and_limit(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    seed(
        store,
        [
            make_record(index=0, team="gtm"),
            make_record(index=1, team="marketing"),
            make_record(index=2, team="marketing"),
        ],
    )

    assert len(store.records(TelemetryQuery(team="marketing"))) == 2
    assert len(store.records(TelemetryQuery(limit=1))) == 1
    store.close()


def test_grouping_by_an_unknown_column_is_refused_rather_than_interpolated(
    tmp_path: Path,
) -> None:
    store = open_store(tmp_path)
    with pytest.raises(ValueError, match="cannot group telemetry by"):
        store.aggregate(None, "team; DROP TABLE telemetry")
    store.close()


def test_rows_with_no_provider_are_still_counted(tmp_path: Path) -> None:
    """A request blocked before routing has no provider but is still traffic."""
    store = open_store(tmp_path)
    seed(store, [make_record(index=0, provider=None, model=None, status="blocked")])

    (row,) = store.aggregate(None, "provider")
    assert row.key == "unknown"
    assert row.requests == 1
    assert row.errors == 1
    store.close()
