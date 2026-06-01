"""Rate limiting for the FinSage-7B API.

Two backends share one contract (:class:`RateLimiter`):

- :class:`InMemoryRateLimiter` — a thread-safe sliding window. Single-process;
  ideal for development and a single API replica.
- :class:`RedisRateLimiter` — a fixed-window counter in Redis, shared across
  replicas. Selected via ``RATE_LIMIT_BACKEND=redis`` + ``REDIS_URL``.

:func:`build_rate_limiter` chooses the backend from settings and falls back to
in-memory (with a warning) if Redis is unavailable, so the service never fails
to start because of the limiter.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from finsage.logging_utils import get_logger

if TYPE_CHECKING:
    from finsage.config import Settings

logger = get_logger(__name__)

#: Length of the rate-limit window in seconds.
WINDOW_SECONDS = 60.0


@runtime_checkable
class RateLimiter(Protocol):
    """Protocol implemented by every rate-limiter backend."""

    def is_allowed(self, client_id: str) -> tuple[bool, dict[str, int]]:
        """Return ``(allowed, metadata)`` for a request by ``client_id``."""
        ...

    def reset(self) -> None:
        """Clear all recorded request history."""
        ...


class InMemoryRateLimiter:
    """A thread-safe, fixed-budget sliding-window rate limiter.

    Each client is allowed ``requests_per_minute`` requests within any rolling
    60-second window.

    Args:
        requests_per_minute: Maximum requests permitted per client per window.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.requests_per_minute = requests_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, dict[str, int]]:
        """Record a request for ``client_id`` and report whether it is allowed.

        Args:
            client_id: Stable identifier for the caller (hashed API key or IP).

        Returns:
            A ``(allowed, metadata)`` tuple. ``metadata`` contains ``limit``,
            ``remaining``, and ``reset_seconds`` (seconds until the window frees
            up capacity).
        """
        now = time.monotonic()
        cutoff = now - WINDOW_SECONDS
        with self._lock:
            hits = self._hits[client_id]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            limit = self.requests_per_minute
            if len(hits) >= limit:
                reset_seconds = int(max(0.0, hits[0] + WINDOW_SECONDS - now)) + 1
                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset_seconds": reset_seconds,
                }

            hits.append(now)
            remaining = max(0, limit - len(hits))
            reset_seconds = int(WINDOW_SECONDS)
            if hits:
                reset_seconds = int(max(0.0, hits[0] + WINDOW_SECONDS - now)) + 1
            return True, {
                "limit": limit,
                "remaining": remaining,
                "reset_seconds": reset_seconds,
            }

    def reset(self) -> None:
        """Clear all recorded request history."""
        with self._lock:
            self._hits.clear()


class RedisRateLimiter:
    """A fixed-window rate limiter backed by Redis (shared across replicas).

    Uses an atomic ``INCR`` on a per-window key with a TTL, so multiple API
    replicas enforce a single shared budget. The ``redis`` client is duck-typed
    (``incr``, ``expire``, ``ttl``, ``flushdb``) so a fake can be injected in
    tests.

    Args:
        client: A Redis client (e.g. ``redis.Redis``).
        requests_per_minute: Maximum requests permitted per client per window.
        namespace: Key prefix for this limiter's counters.
    """

    def __init__(
        self, client: Any, requests_per_minute: int = 60, namespace: str = "finsage:ratelimit"
    ) -> None:
        self._client = client
        self.requests_per_minute = requests_per_minute
        self.namespace = namespace

    def is_allowed(self, client_id: str) -> tuple[bool, dict[str, int]]:
        """Increment the current window counter and report whether it is allowed.

        Args:
            client_id: Stable identifier for the caller (hashed API key or IP).

        Returns:
            A ``(allowed, metadata)`` tuple with ``limit``, ``remaining``, and
            ``reset_seconds``.
        """
        window = int(time.time() // WINDOW_SECONDS)
        key = f"{self.namespace}:{client_id}:{window}"
        limit = self.requests_per_minute

        count = int(self._client.incr(key))
        if count == 1:
            # First hit in this window — set the expiry so the key self-cleans.
            self._client.expire(key, int(WINDOW_SECONDS))

        ttl = int(self._client.ttl(key))
        reset_seconds = ttl if ttl and ttl > 0 else int(WINDOW_SECONDS)
        allowed = count <= limit
        remaining = max(0, limit - count)
        return allowed, {"limit": limit, "remaining": remaining, "reset_seconds": reset_seconds}

    def reset(self) -> None:
        """Clear all counters (test/maintenance helper)."""
        flush = getattr(self._client, "flushdb", None)
        if callable(flush):
            flush()


def build_rate_limiter(settings: Settings) -> RateLimiter:
    """Build the configured rate limiter, falling back to in-memory on failure.

    Args:
        settings: Application settings (``rate_limit_backend``, ``redis_url``,
            ``rate_limit_requests_per_minute``).

    Returns:
        A :class:`RateLimiter`. Returns an :class:`InMemoryRateLimiter` when the
        backend is ``memory`` or when Redis is requested but unavailable.
    """
    rpm = settings.rate_limit_requests_per_minute
    backend = settings.rate_limit_backend.strip().lower()
    if backend != "redis":
        return InMemoryRateLimiter(rpm)

    try:
        import redis  # type: ignore[import-untyped]

        if not settings.redis_url:
            raise ValueError("REDIS_URL is required when RATE_LIMIT_BACKEND=redis")
        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        logger.info("Rate limiting via Redis at %s", settings.redis_url)
        return RedisRateLimiter(client, rpm)
    except Exception as exc:  # noqa: BLE001 — never let the limiter break startup
        logger.warning(
            "Redis rate-limit backend unavailable (%s); falling back to in-memory. "
            "In-memory limiting is per-process only.",
            exc,
        )
        return InMemoryRateLimiter(rpm)
