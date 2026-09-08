"""AC-17 end to end: `docker compose up` → `/healthz` → a completion via mock.

Two halves, because the claim has two halves.

The **static** half reads `docker-compose.yml` and `infra/Dockerfile` and asserts
the properties that make a zero-key run possible at all: no service is handed a
vendor credential, the gateway does not wait on the model download, the image
declares a non-root user and a healthcheck. These run everywhere, including on a
machine with no Docker, because they are assertions about the files.

The **live** half runs against a stack that is actually up. It is skipped unless
`CONDUIT_E2E_BASE_URL` names one, which keeps `pytest` runnable offline while
letting CI's container job — and anyone who has just typed `docker compose up
-d` — get the real assertion:

    docker compose up -d --wait gateway
    CONDUIT_E2E_BASE_URL=http://localhost:8000 pytest tests/test_compose_e2e.py

`mock:echo` is pinned on the request rather than left to the router. The
contract reads a set `model` as a request rather than a hint, so pinning is what
makes "a completion via mock" a deterministic assertion instead of a bet on
whether Ollama happens to have been seeded on this machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"
DOCKERFILE = ROOT / "infra" / "Dockerfile"

#: The demo key shipped in `config/gateway.yaml`. Not a secret — it is the
#: credential a reviewer is meant to use on a stack that has no others.
DEMO_KEY = "conduit-demo-key"

BASE_URL_ENV = "CONDUIT_E2E_BASE_URL"

#: Anything that looks like it carries a model vendor's credential. A compose
#: file that mentions one has broken the zero-key claim even if the value is
#: empty, because it means the stack expects the caller to supply it.
CREDENTIAL_MARKERS = ("API_KEY", "CREDENTIALS", "SECRET", "TOKEN")

live = pytest.mark.skipif(
    not os.environ.get(BASE_URL_ENV),
    reason=f"no running stack; set {BASE_URL_ENV} after `docker compose up -d --wait gateway`",
)


def _compose() -> dict[str, Any]:
    parsed: Any = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), "docker-compose.yml must be a mapping"
    return parsed


@pytest.fixture(scope="module")
def client() -> Any:
    base = os.environ.get(BASE_URL_ENV, "")
    with httpx.Client(base_url=base, timeout=60.0) as session:
        yield session


# --------------------------------------------------------------------------- #
# Static: the compose file and image can satisfy AC-17
# --------------------------------------------------------------------------- #


def test_no_service_is_handed_a_vendor_credential() -> None:
    """The stack must not require, or silently forward, an API key."""
    services: dict[str, Any] = _compose()["services"]
    offending: list[str] = []
    for name, service in services.items():
        for key in service.get("environment") or {}:
            if any(marker in str(key).upper() for marker in CREDENTIAL_MARKERS):
                offending.append(f"{name}.{key}")
    assert not offending, (
        f"docker-compose.yml passes credential-shaped env vars {offending}. AC-17 requires "
        "`docker compose up` to work with zero API keys set, which means the stack cannot "
        "name one. Add keys through the optional .env file instead."
    )


def test_the_gateway_does_not_wait_on_the_model_download() -> None:
    """Seeding Ollama is gigabytes; `up` must not block on it.

    `ollama-pull` may exist and may be profiled off, but it must never be a
    `depends_on` of the gateway — that would make AC-17 depend on link speed.
    """
    compose = _compose()
    gateway = compose["services"]["gateway"]
    assert "ollama-pull" not in (gateway.get("depends_on") or {})
    # And the thing it *does* wait on must not itself gate on a pull.
    assert set(gateway.get("depends_on") or {}) <= {"ollama"}


def test_the_image_runs_as_a_non_root_user_with_a_healthcheck() -> None:
    """AC-2: non-root, and a healthcheck that reports something."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    users = [
        line.split(maxsplit=1)[1].strip()
        for line in dockerfile.splitlines()
        if line.startswith("USER ")
    ]
    assert users, "infra/Dockerfile never switches away from root"
    assert users[-1] not in ("root", "0", "0:0"), f"final USER is {users[-1]!r}"
    assert "HEALTHCHECK" in dockerfile, "infra/Dockerfile declares no healthcheck"


# --------------------------------------------------------------------------- #
# Live: a stack that is actually running
# --------------------------------------------------------------------------- #


@live
def test_healthz_reports_ok(client: httpx.Client) -> None:
    """The liveness probe the container healthcheck and Cloud Run both use."""
    response = client.get("/healthz")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}


@live
def test_readyz_lists_only_the_keyless_providers(client: httpx.Client) -> None:
    """With no keys set, the live provider set is exactly mock (+ Ollama).

    This is the observable form of the registry's dormancy rule: an absent
    `ANTHROPIC_API_KEY` does not merely disable the adapter, it removes every
    model that adapter served from routing.
    """
    body = client.get("/readyz").json()
    names = {entry["name"] for entry in body["providers"]}
    assert "mock" in names
    assert not names - {"mock", "ollama"}, f"a keyed provider is live: {sorted(names)}"
    assert any(entry["healthy"] for entry in body["providers"])


@live
def test_a_completion_comes_back_through_mock(client: httpx.Client) -> None:
    """The AC-17 assertion proper: a real completion, zero keys, via mock."""
    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": DEMO_KEY},
        json={
            "model": "mock:echo",
            "messages": [{"role": "user", "content": "Is Acme Corp enterprise or SMB?"}],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["model"] == "mock:echo"
    assert body["x_conduit"]["provider"] == "mock"
    assert body["choices"][0]["message"]["content"], "mock returned an empty completion"
    # Cost accounting still runs on the free path; it just prices at zero.
    assert body["usage"]["total_tokens"] > 0
    assert body["x_conduit"]["cost_usd"] == 0.0


@live
def test_an_unauthenticated_request_is_refused(client: httpx.Client) -> None:
    """A zero-key stack is not an open proxy: `require_auth` is still on."""
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 401, response.text
