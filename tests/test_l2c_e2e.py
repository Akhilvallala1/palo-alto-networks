"""End-to-end: the workflow against a real gateway, plus the shipped config.

`run_quote` here goes over HTTP into the real FastAPI application — auth, both
guards, the router, the failover chain and the telemetry recorder all run. The
transport is `httpx.ASGITransport`, so it is real HTTP semantics through the
real stack with no socket and no network, which is what lets this run in CI.

Issue #9's E2E criterion names `docker compose up` as the harness. The compose
file belongs to issue #10, which is not merged, so the container path is not
exercised here; what is exercised is the gateway it would be running.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from apps.l2c.client import ConduitClient
from apps.l2c.graph import run_quote
from apps.l2c.rag import build_index
from conduit.config import load_config
from conduit.contracts import Complexity
from conduit.providers.registry import available_providers, build_registry
from conduit.telemetry.store import GROUPABLE_COLUMNS
from tests.gateway_fixtures import GOOD_KEY, build_app
from tests.l2c_fixtures import make_request

REPO = Path(__file__).resolve().parent.parent
APPS = REPO / "apps"
CONFIG_DIR = REPO / "config"

# Same set the AC-16 guard uses for src/; issue #9 AC-5 extends the rule to apps/.
VENDOR_ROOTS = {
    "anthropic",
    "openai",
    "google",
    "cohere",
    "mistralai",
    "boto3",
    "vertexai",
}


@pytest.fixture
def gateway(tmp_path):
    """The real app, with a real recorder writing into a temp directory."""
    return build_app(tmp_path)


async def test_quote_runs_through_the_real_gateway_and_writes_telemetry(gateway) -> None:
    """AC-1, AC-2 and AC-4 over the real HTTP path, asserted against the store."""
    index = build_index()
    try:
        async with ConduitClient(app=gateway, api_key=GOOD_KEY) as client:
            decision = await run_quote(make_request(discount_pct=26.0), client=client, index=index)

        assert decision.outcome == "escalate"
        # The citation resolves to a real chunk — asserted, not eyeballed.
        section = index.resolve(decision.citation)
        assert section is not None
        assert decision.citation == "DISC-4.3"
        assert decision.citation_quote == section.text
    finally:
        index.close()

    records = gateway.state.recorder.store.records()

    # One row per model call, and the router node made none.
    assert len(records) == 4
    assert len({record.trace_id for record in records}) == 4
    assert all(record.status == "ok" for record in records)

    by_workflow = {record.workflow: record.tier for record in records if record.workflow}
    assert by_workflow["intake"] is Complexity.TRIVIAL
    assert by_workflow["discount_analyst"] is Complexity.COMPLEX
    # AC-4: the cheap node and the expensive node did not land on one tier.
    assert by_workflow["intake"] is not by_workflow["discount_analyst"]
    assert len(set(by_workflow.values())) >= 2

    # The columns the demo's cost story is told with are the groupable ones.
    assert {"workflow", "tier"} <= GROUPABLE_COLUMNS


async def test_the_workflow_runs_with_zero_api_keys(gateway, monkeypatch) -> None:
    """AC-7: with every credential removed, the run still completes."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Asserted rather than assumed: no hosted adapter can even be constructed.
    assert not {"anthropic", "openai", "gemini"} & set(available_providers())

    index = build_index()
    try:
        async with ConduitClient(app=gateway, api_key=GOOD_KEY) as client:
            decision = await run_quote(
                make_request(discount_pct=8.0, partner_involved=False),
                client=client,
                index=index,
            )
    finally:
        index.close()

    assert decision.outcome == "approve"
    assert decision.citation == "DISC-4.1"
    # Every call was served by an offline adapter.
    assert {call.provider for call in decision.calls} <= {"mock", "scripted"}


def test_the_shipped_routing_config_leaves_every_tier_callable_with_no_keys() -> None:
    """The zero-key path is a property of config/routing.yaml, so assert it there.

    A hosted model drops out of the registry when its key is absent, so without a
    terminal offline entry a tier's chain can empty out and AC-7 fails at the
    router rather than anywhere obvious.
    """
    config = load_config(CONFIG_DIR, env={})
    registry = build_registry(models=config.models, env={})

    for tier in (Complexity.TRIVIAL, Complexity.STANDARD, Complexity.COMPLEX):
        chain = registry.usable_chain(config.routing.tiers[tier], tier)
        assert chain, f"{tier.value} tier has no callable model with zero keys"


@pytest.mark.parametrize("path", sorted(APPS.rglob("*.py")), ids=lambda p: str(p.relative_to(APPS)))
def test_no_file_under_apps_imports_a_vendor_sdk(path) -> None:
    """AC-5. Parsed, not grepped, so a nested-scope import is caught too."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    offending = roots & VENDOR_ROOTS
    assert not offending, (
        f"{path.relative_to(REPO)} imports vendor SDK(s) {sorted(offending)}. "
        "The L2C app reaches models only through Conduit's HTTP API."
    )


def test_the_apps_tree_was_actually_scanned() -> None:
    """Guards the parametrisation above against silently collecting nothing."""
    assert len(list(APPS.rglob("*.py"))) >= 5


def test_seeded_gtm_data_is_deterministic(tmp_path) -> None:
    """AC-6: same seed, byte-identical output; different seed, different output."""
    script = REPO / "scripts" / "seed_gtm_data.py"

    def run(out: Path, seed: int) -> None:
        result = subprocess.run(
            [sys.executable, str(script), "--out", str(out), "--seed", str(seed)],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        assert result.returncode == 0, result.stderr

    first, second, other = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    run(first, 20260908)
    run(second, 20260908)
    run(other, 99)

    names = sorted(p.name for p in first.glob("*.json"))
    assert names, "the seeder wrote no files"
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    assert (first / "accounts.json").read_bytes() != (other / "accounts.json").read_bytes()


def test_seeded_data_is_labelled_synthetic(tmp_path) -> None:
    """The corpus is authored, not customer data; the manifest has to say so."""
    script = REPO / "scripts" / "seed_gtm_data.py"
    out = tmp_path / "gtm"
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(out), "--seed", "20260908"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True

    # Contacts use RFC 2606 reserved domains, so nothing here can reach a real inbox.
    contacts = json.loads((out / "contacts.json").read_text(encoding="utf-8"))
    assert contacts
    assert all(contact["email"].endswith(".example") for contact in contacts)
