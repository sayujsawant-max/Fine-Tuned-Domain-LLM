"""Tests for the in-memory rate limiter."""

from __future__ import annotations

from finsage.serving.rate_limiter import InMemoryRateLimiter


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
