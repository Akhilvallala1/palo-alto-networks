"""FastAPI gateway: auth, rate limiting, and the OpenAI-shaped routes.

The integration point. Providers (#3), router (#4), guards (#5) and telemetry
(#7) are built independently against `conduit.contracts`; this package is where
they are wired into one HTTP surface and where the seams between them are
resolved.

    from conduit.gateway import create_app
    app = create_app()

Run it with `uvicorn conduit.gateway.app:create_app --factory`.
"""

from .app import configure_logging, create_app
from .auth import ApiKeyDirectory, Principal
from .errors import GatewayError, GuardRejectedError, RateLimitExceededError, UnauthorizedError
from .pipeline import PIPELINE_ORDER, Pipeline, PipelineOrderError, PipelineOutcome, Stage
from .ratelimit import RateLimitDecision, RateLimiter
from .settings import ApiKeySettings, GatewaySettings, load_gateway

__all__ = [
    "PIPELINE_ORDER",
    "ApiKeyDirectory",
    "ApiKeySettings",
    "GatewayError",
    "GatewaySettings",
    "GuardRejectedError",
    "Pipeline",
    "PipelineOrderError",
    "PipelineOutcome",
    "Principal",
    "RateLimitDecision",
    "RateLimitExceededError",
    "RateLimiter",
    "Stage",
    "UnauthorizedError",
    "configure_logging",
    "create_app",
    "load_gateway",
]
