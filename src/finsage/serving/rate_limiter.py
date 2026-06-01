"""In-memory sliding-window rate limiter.

Single-process only — suitable for development and a single API replica. For
multi-replica production deployments, swap in a shared store (e.g. Redis); the
:meth:`InMemoryRateLimiter.is_allowed` contract is designed to make that
substitution straightforward.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

#: Length of the rate-limit window in seconds.
WINDOW_SECONDS = 60.0


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
