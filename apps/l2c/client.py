"""The app's only route to a model: Conduit's HTTP API.

Nothing under `apps/` imports a vendor SDK, and this module is why that costs
nothing to comply with. It speaks HTTP to `POST /v1/complete` — the native
route, which takes and returns the frozen contract shapes, so there is no
OpenAI-flavoured translation layer to keep honest in both directions.

Two transports, one code path:

* **Remote.** `ConduitClient(base_url=...)` against a gateway that is already
  running, which is what `docker compose up` and the CLI's `--gateway-url` use.
* **Embedded.** `ConduitClient(app=create_app())` drives the real FastAPI
  application in-process over `httpx.ASGITransport`. The request goes through
  auth, both guards, the router, the failover chain and the telemetry recorder
  exactly as a socket request would; it simply never reaches a socket. That is
  what lets the demo run with zero setup and the tests run with no network.

The `workflow` argument is not decoration. It becomes `metadata["workflow"]`,
which the router reads as a tier hint and the telemetry recorder stores as a
column — so it is simultaneously the thing that routes a node and the thing
that makes per-node cost queryable afterwards.
"""

from types import TracebackType

import httpx

from conduit.contracts import CompletionRequest, CompletionResponse, Message

__all__ = ["ConduitClient", "ConduitClientError"]

#: The demo key shipped in `config/gateway.yaml`. Not a secret: it exists so the
#: zero-key path still exercises the gateway's real auth stage.
DEFAULT_API_KEY = "conduit-demo-key"

DEFAULT_TIMEOUT_S = 60.0


class ConduitClientError(RuntimeError):
    """The gateway refused or failed a request.

    Carries the status code and body, because the interesting failures here are
    a 400 from a guard and a 429 from the budget engine, and both say why.
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"conduit returned HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class ConduitClient:
    """An async client for one Conduit gateway."""

    def __init__(
        self,
        *,
        base_url: str = "http://conduit.local",
        app: object | None = None,
        api_key: str = DEFAULT_API_KEY,
        team: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        transport: httpx.AsyncBaseTransport | None = None
        if app is not None:
            # `raise_app_exceptions=False` would swallow a gateway bug into a
            # 500 the caller cannot debug; letting it propagate is better in a
            # demo app whose failures are all local.
            transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        self._client = httpx.AsyncClient(
            transport=transport,
            base_url=base_url,
            timeout=timeout_s,
            headers={"X-API-Key": api_key},
        )
        self._team = team

    async def complete(
        self,
        *,
        workflow: str,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> CompletionResponse:
        """One completion, tagged with the workflow that asked for it.

        `model` is deliberately left unset: the router's whole job is to pick a
        tier, and pinning a model here would route around the thing the demo
        exists to show.
        """
        metadata = {"workflow": workflow}
        if self._team is not None:
            metadata["team"] = self._team
        request = CompletionRequest(
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=user),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            metadata=metadata,
        )
        response = await self._client.post("/v1/complete", json=request.model_dump(mode="json"))
        if response.status_code != httpx.codes.OK:
            raise ConduitClientError(response.status_code, response.text)
        return CompletionResponse.model_validate(response.json())

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ConduitClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
