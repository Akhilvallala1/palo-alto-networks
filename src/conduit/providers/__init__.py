"""Provider abstraction: adapters implementing `conduit.contracts.Provider`.

This is the only package permitted to import a vendor SDK (epic AC-16), and in
practice it imports none: every adapter is written against the vendor's HTTP
API with an injectable `httpx` transport, so the SDKs stay optional extras and
CI never needs a key or a socket.

What lives here:

- `base` — retry, timeout, HTTP error classification, usage pricing.
- `anthropic`, `ollama`, `mock` — always available adapters.
- `openai`, `gemini` — written, dormant unless their key env var is set.
- `registry` — model id -> provider, price, context window, tier eligibility.
- `failover` — ordered chain per tier, with a per-provider circuit breaker.
"""

from .anthropic import AnthropicProvider
from .base import (
    BaseProvider,
    HttpProvider,
    ProviderError,
    ProviderRefusedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitedError,
    RawCompletion,
    RetryPolicy,
    price_usage,
)
from .failover import (
    AllProvidersFailedError,
    BreakerState,
    CircuitBreaker,
    CircuitBreakers,
    CircuitOpenError,
    FailoverChain,
    FailoverEvent,
)
from .gemini import GeminiProvider
from .mock import MockProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .registry import (
    ModelRegistry,
    UnknownModelError,
    available_providers,
    build_providers,
    build_registry,
)

__all__ = [
    "AllProvidersFailedError",
    "AnthropicProvider",
    "BaseProvider",
    "BreakerState",
    "CircuitBreaker",
    "CircuitBreakers",
    "CircuitOpenError",
    "FailoverChain",
    "FailoverEvent",
    "GeminiProvider",
    "HttpProvider",
    "MockProvider",
    "ModelRegistry",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "ProviderRefusedError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RateLimitedError",
    "RawCompletion",
    "RetryPolicy",
    "UnknownModelError",
    "available_providers",
    "build_providers",
    "build_registry",
    "price_usage",
]
