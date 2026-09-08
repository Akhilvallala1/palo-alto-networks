"""The HTTP surface: two completion routes and two health probes.

`/v1/chat/completions` and `/v1/complete` are the same pipeline behind two
schemas — the OpenAI one for drop-in clients, and `contracts` directly for
callers that would rather not round-trip through someone else's shape. Neither
route contains pipeline logic; both call `Pipeline.complete` and translate.

Auth is a FastAPI dependency rather than middleware, which is what makes AC-3
structural: the route body cannot run before its dependencies resolve, so a bad
key cannot reach a guard or a provider no matter what the handler does.
"""

import asyncio
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, Response, status
from pydantic import BaseModel

from conduit.contracts import CompletionRequest, CompletionResponse
from conduit.gateway.auth import ApiKeyDirectory, Principal, extract_key
from conduit.gateway.errors import RateLimitExceededError
from conduit.gateway.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    to_completion_request,
    to_openai_response,
)
from conduit.gateway.pipeline import Pipeline
from conduit.gateway.ratelimit import RateLimiter
from conduit.providers.registry import ModelRegistry

__all__ = [
    "HealthResponse",
    "ProviderHealth",
    "ReadinessResponse",
    "create_routes",
]


class ProviderHealth(BaseModel):
    name: str
    healthy: bool


class HealthResponse(BaseModel):
    """`/healthz`: is this process running. Deliberately dependency-free."""

    status: str = "ok"


class ReadinessResponse(BaseModel):
    """`/readyz`: can this process actually serve traffic right now."""

    status: str
    providers: list[ProviderHealth]
    models: int


def create_routes(
    *,
    pipeline: Pipeline,
    directory: ApiKeyDirectory,
    limiter: RateLimiter,
    registry: ModelRegistry,
) -> APIRouter:
    """Build the router with its collaborators closed over.

    Closure rather than module-level globals: `create_app` can then build two
    independently-configured apps in one process, which is what lets the tests
    run a guard-blocking gateway and a permissive one side by side.
    """
    router = APIRouter()

    async def authenticate(
        x_api_key: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Principal:
        """Stage 1. Runs before any handler body, so before any guard work."""
        return directory.resolve(extract_key(x_api_key, authorization))

    async def admit(
        principal: Annotated[Principal, Depends(authenticate)],
    ) -> Principal:
        """Rate-limit admission, inside the auth stage and before the pipeline."""
        decision = limiter.check(principal.fingerprint, principal.rate_limit_per_minute)
        if not decision.allowed:
            raise RateLimitExceededError(decision.retry_after_s, principal.rate_limit_per_minute)
        return principal

    # A type alias, hence the class-style name: every route that needs a caller
    # depends on `admit`, so auth and rate limiting cannot be forgotten on a new
    # route by omission.
    Caller = Annotated[Principal, Depends(admit)]  # noqa: N806

    @router.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        response_model_exclude_none=True,
        summary="OpenAI-shaped completion (drop-in for OpenAI SDK clients)",
        tags=["completions"],
    )
    async def chat_completions(
        payload: ChatCompletionRequest,
        principal: Caller,
        response: Response,
    ) -> ChatCompletionResponse:
        req = to_completion_request(payload)
        outcome = await pipeline.complete(req, principal)
        completion = outcome.unwrap()
        response.headers.update(outcome.headers)
        return to_openai_response(completion, trace_id=outcome.trace_id, max_tokens=req.max_tokens)

    @router.post(
        "/v1/complete",
        response_model=CompletionResponse,
        summary="Native Conduit completion, using the frozen contract models",
        tags=["completions"],
    )
    async def complete(
        payload: CompletionRequest,
        principal: Caller,
        response: Response,
    ) -> CompletionResponse:
        """The contract in, the contract out — no translation either way.

        The response is returned exactly as the failover chain produced it,
        `latency_ms` included. That field is the provider's own measurement of
        its own call; the caller-visible wall clock lives in telemetry's
        `latency_ms` column and will legitimately be the larger number.
        """
        outcome = await pipeline.complete(payload, principal)
        completion = outcome.unwrap()
        response.headers.update(outcome.headers)
        return completion

    @router.get("/healthz", response_model=HealthResponse, tags=["health"])
    async def healthz() -> HealthResponse:
        """Liveness. Never touches a provider, so it cannot fail on a vendor."""
        return HealthResponse()

    @router.get("/readyz", response_model=ReadinessResponse, tags=["health"])
    async def readyz(response: Response) -> ReadinessResponse:
        """Readiness, including per-provider reachability (AC-7).

        Probes run concurrently: readiness should cost one provider timeout, not
        the sum of them. `Provider.health` never raises by contract, but
        `return_exceptions` keeps one misbehaving adapter from taking the probe
        down with it.
        """
        providers = registry.providers
        results = await asyncio.gather(
            *(provider.health() for provider in providers), return_exceptions=True
        )
        health = [
            ProviderHealth(name=provider.name, healthy=result is True)
            for provider, result in zip(providers, results, strict=True)
        ]
        ready = any(entry.healthy for entry in health)
        if not ready:
            # No provider can be reached: the gateway is up but cannot serve.
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            structlog.get_logger("conduit.gateway").error(
                "gateway.not_ready", providers=[entry.name for entry in health]
            )
        return ReadinessResponse(
            status="ready" if ready else "unavailable",
            providers=health,
            models=len(registry),
        )

    return router
