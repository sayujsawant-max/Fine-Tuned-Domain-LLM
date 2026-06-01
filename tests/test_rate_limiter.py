"""Tests for the in-memory rate limiter."""

from __future__ import annotations

from finsage.config import Settings
from finsage.serving.rate_limiter import (
    InMemoryRateLimiter,
    RateLimiter,
    RedisRateLimiter,
    build_rate_limiter,
)


def test_allows_under_limit():
    """Requests under the limit are allowed and decrement remaining."""
    limiter = InMemoryRateLimiter(requests_per_minute=3)
    allowed, meta = limiter.is_allowed("client")
    assert allowed is True
    assert meta["limit"] == 3
    assert meta["remaining"] == 2


def test_blocks_over_limit():
    """Requests beyond the limit are blocked with remaining 0."""
    limiter = InMemoryRateLimiter(requests_per_minute=2)
    assert limiter.is_allowed("c")[0] is True
    assert limiter.is_allowed("c")[0] is True
    allowed, meta = limiter.is_allowed("c")
    assert allowed is False
    assert meta["remaining"] == 0
    assert meta["reset_seconds"] >= 1


def test_independent_clients():
    """Different clients have independent budgets."""
    limiter = InMemoryRateLimiter(requests_per_minute=1)
    assert limiter.is_allowed("a")[0] is True
    assert limiter.is_allowed("b")[0] is True
    assert limiter.is_allowed("a")[0] is False


def test_reset_clears_history():
    """reset() restores full capacity."""
    limiter = InMemoryRateLimiter(requests_per_minute=1)
    assert limiter.is_allowed("c")[0] is True
    assert limiter.is_allowed("c")[0] is False
    limiter.reset()
    assert limiter.is_allowed("c")[0] is True


def test_metadata_keys_present():
    """Metadata exposes limit, remaining, and reset_seconds."""
    _, meta = InMemoryRateLimiter(requests_per_minute=5).is_allowed("c")
    assert set(meta) == {"limit", "remaining", "reset_seconds"}


def test_in_memory_satisfies_protocol():
    """InMemoryRateLimiter is a RateLimiter."""
    assert isinstance(InMemoryRateLimiter(), RateLimiter)


def test_build_rate_limiter_defaults_to_memory():
    """The factory returns an in-memory limiter when backend=memory."""
    settings = Settings(_env_file=None, RATE_LIMIT_BACKEND="memory")
    limiter = build_rate_limiter(settings)
    assert isinstance(limiter, InMemoryRateLimiter)


def test_build_rate_limiter_falls_back_when_redis_unavailable():
    """Requesting redis without a reachable server falls back to in-memory."""
    settings = Settings(
        _env_file=None,
        RATE_LIMIT_BACKEND="redis",
        REDIS_URL="redis://127.0.0.1:6390/0",  # nothing listening
    )
    limiter = build_rate_limiter(settings)
    assert isinstance(limiter, InMemoryRateLimiter)


class _FakeRedis:
    """Minimal in-process fake of the redis client commands we use."""

    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    def expire(self, key: str, ttl: int) -> bool:
        return True

    def ttl(self, key: str) -> int:
        return 42

    def flushdb(self) -> None:
        self.store.clear()


def test_redis_rate_limiter_allows_then_blocks():
    """RedisRateLimiter blocks once the per-window count exceeds the limit."""
    limiter = RedisRateLimiter(_FakeRedis(), requests_per_minute=2)
    assert limiter.is_allowed("c")[0] is True
    assert limiter.is_allowed("c")[0] is True
    allowed, meta = limiter.is_allowed("c")
    assert allowed is False
    assert meta["limit"] == 2
    assert meta["remaining"] == 0
    assert meta["reset_seconds"] == 42
