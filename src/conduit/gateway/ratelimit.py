"""Per-key request rate limiting.

`mcp-gateway`'s sliding window, kept deliberately: a deque of hit timestamps per
key, trimmed on read. It is in-memory and therefore per-replica — correct for a
single gateway process, and the thing to swap for Redis before running two.

The one addition is `retry_after_s`. AC-4 requires a `Retry-After` header, and a
sliding window can compute an honest one: the oldest hit in the window expires
at `oldest + 60s`, and that is the first instant the caller would be admitted.
A fixed window cannot say that without lying by up to a full window.

Rate limiting is admission control on an authenticated key, so it runs inside
the auth stage — after the key resolves, before any guard or provider work.
"""

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

__all__ = ["RateLimitDecision", "RateLimiter"]


@dataclass(frozen=True)
class RateLimitDecision:
    """The verdict plus the numbers a caller needs to back off intelligently."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_s: int

    def headers(self) -> dict[str, str]:
        """`X-RateLimit-*` on every answer, `Retry-After` only on a refusal."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if not self.allowed:
            headers["Retry-After"] = str(self.retry_after_s)
        return headers


class RateLimiter:
    """Sliding-window limiter keyed by API key.

    `clock` is injectable so a test can cross a window boundary without
    sleeping for a minute.
    """

    def __init__(
        self,
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int) -> RateLimitDecision:
        """Admit or refuse one request, recording it when admitted.

        The limit is passed per call rather than held on the limiter because it
        comes from the key's own row in `gateway.yaml`; two keys with different
        quotas share one limiter.
        """
        if limit < 1:
            raise ValueError(f"rate limit must be at least 1 request/window, got {limit}")

        now = self._clock()
        hits = self._hits[key]
        cutoff = now - self._window_s
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= limit:
            # The window frees a slot when its oldest hit falls out of it.
            retry_after = max(1, ceil(hits[0] + self._window_s - now))
            return RateLimitDecision(
                allowed=False, limit=limit, remaining=0, retry_after_s=retry_after
            )

        hits.append(now)
        return RateLimitDecision(
            allowed=True, limit=limit, remaining=max(0, limit - len(hits)), retry_after_s=0
        )

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)
