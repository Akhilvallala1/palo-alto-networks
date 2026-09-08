"""Unit tests for percentile math, slicing, and the /metrics + /dashboard views."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conduit.contracts import Complexity
from conduit.telemetry.metrics import (
    Aggregate,
    collect,
    create_metrics_router,
    percentile,
    slice_by,
)
from conduit.telemetry.store import JsonlSqliteStore, TelemetryQuery
from telemetry_fixtures import make_record, open_store, seed

# 10 requests with hand-checkable slices:
#   anthropic / gtm       / trivial  x3  @ $0.001, 100ms
#   anthropic / gtm       / complex  x2  @ $0.010, 300ms
#   ollama    / marketing / standard x5  @ $0.000,  50ms
FIXTURE = [
    *(
        make_record(index=i, tier=Complexity.TRIVIAL, cost_usd=0.001, latency_ms=100)
        for i in range(3)
    ),
    *(
        make_record(
            index=10 + i,
            tier=Complexity.COMPLEX,
            model="claude-opus-5",
            cost_usd=0.010,
            latency_ms=300,
        )
        for i in range(2)
    ),
    *(
        make_record(
            index=20 + i,
            team="marketing",
            workflow="campaign_copy",
            tier=Complexity.STANDARD,
            provider="ollama",
            model="ollama:llama3.2",
            cost_usd=0.0,
            latency_ms=50,
        )
        for i in range(5)
    ),
]


def seeded_store(tmp_path: Path) -> JsonlSqliteStore:
    store = open_store(tmp_path)
    seed(store, FIXTURE)
    return store


def by_key(rows: list[Aggregate]) -> dict[str, Aggregate]:
    return {row.key: row for row in rows}


# --------------------------------------------------------------------------- #
# Percentile math
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("samples", "p", "expected"),
    [
        # 10, 20, ... 200: nearest rank puts p50 at the 10th sample and p95 at the 19th.
        ([n * 10 for n in range(1, 21)], 50, 100),
        ([n * 10 for n in range(1, 21)], 95, 190),
        ([5], 50, 5),
        ([5], 95, 5),
        ([3, 1, 2], 50, 2),  # unsorted input is sorted first
        ([1, 2, 3, 4], 100, 4),
        ([], 50, 0),
    ],
)
def test_percentile_is_nearest_rank(samples: list[int], p: float, expected: int) -> None:
    assert percentile(samples, p) == expected


def test_percentile_rejects_a_meaningless_quantile() -> None:
    with pytest.raises(ValueError, match="percentile must be in"):
        percentile([1, 2, 3], 0)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_overall_aggregate_matches_hand_computed_totals(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    overall = collect(store).overall

    assert overall.requests == 10
    assert overall.cost_usd == pytest.approx(3 * 0.001 + 2 * 0.010)
    assert overall.prompt_tokens == 10 * 100
    assert overall.completion_tokens == 10 * 50
    assert overall.total_tokens == 10 * 150
    assert overall.errors == 0
    store.close()


def test_overall_percentiles_match_a_hand_computed_percentile(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    overall = collect(store).overall

    # Sorted latencies: 50 x5, 100 x3, 300 x2.
    # p50 -> ceil(0.50*10) = 5th sample = 50. p95 -> ceil(0.95*10) = 10th = 300.
    assert overall.p50_latency_ms == 50
    assert overall.p95_latency_ms == 300
    store.close()


def test_slicing_by_provider_team_and_tier_matches_the_fixture(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    snapshot = collect(store)

    providers = by_key(snapshot.by_provider)
    assert providers["anthropic"].requests == 5
    assert providers["anthropic"].cost_usd == pytest.approx(0.023)
    assert providers["ollama"].requests == 5
    assert providers["ollama"].cost_usd == 0.0

    teams = by_key(snapshot.by_team)
    assert teams["gtm"].requests == 5
    assert teams["marketing"].requests == 5
    assert teams["marketing"].cost_usd == 0.0

    tiers = by_key(snapshot.by_tier)
    assert {key: row.requests for key, row in tiers.items()} == {
        "trivial": 3,
        "complex": 2,
        "standard": 5,
    }
    assert tiers["complex"].cost_usd == pytest.approx(0.020)
    store.close()


def test_per_slice_percentiles_are_computed_within_the_slice(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    tiers = by_key(slice_by(store, "tier"))

    assert (tiers["trivial"].p50_latency_ms, tiers["trivial"].p95_latency_ms) == (100, 100)
    assert (tiers["complex"].p50_latency_ms, tiers["complex"].p95_latency_ms) == (300, 300)
    assert (tiers["standard"].p50_latency_ms, tiers["standard"].p95_latency_ms) == (50, 50)
    store.close()


def test_filters_narrow_every_slice_at_once(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    snapshot = collect(store, TelemetryQuery(team="gtm"))

    assert snapshot.overall.requests == 5
    assert by_key(snapshot.by_provider).keys() == {"anthropic"}
    assert snapshot.by_team[0].key == "gtm"
    store.close()


def test_an_empty_store_reports_zeroes_rather_than_failing(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    snapshot = collect(store)

    assert snapshot.overall.requests == 0
    assert snapshot.overall.cost_usd == 0.0
    assert snapshot.overall.p95_latency_ms == 0
    assert snapshot.by_provider == []
    store.close()


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #


def client_for(store: JsonlSqliteStore) -> TestClient:
    app = FastAPI()
    app.include_router(create_metrics_router(store))
    return TestClient(app)


def test_metrics_endpoint_exposes_count_tokens_cost_and_percentiles(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    response = client_for(store).get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall"]["requests"] == 10
    assert payload["overall"]["cost_usd"] == pytest.approx(0.023)
    assert payload["overall"]["p50_latency_ms"] == 50
    assert payload["overall"]["p95_latency_ms"] == 300
    assert {row["key"] for row in payload["by_provider"]} == {"anthropic", "ollama"}
    assert {row["key"] for row in payload["by_tier"]} == {"trivial", "standard", "complex"}
    store.close()


def test_metrics_endpoint_honours_slice_filters(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    payload = client_for(store).get("/metrics", params={"provider": "ollama"}).json()

    assert payload["overall"]["requests"] == 5
    assert payload["overall"]["cost_usd"] == 0.0
    store.close()


def test_dashboard_renders_real_rows_with_no_javascript(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    response = client_for(store).get("/dashboard")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "Conduit telemetry" in body
    assert "anthropic" in body and "marketing" in body and "complex" in body
    assert "$0.0230" in body  # overall cost, formatted
    assert "<script" not in body.lower()
    assert "src=" not in body.lower()  # nothing to fetch, no build step
    store.close()


def test_dashboard_survives_an_empty_store(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    response = client_for(store).get("/dashboard")

    assert response.status_code == 200
    assert "no requests recorded" in response.text
    store.close()
