"""Tests for vLLM health helpers (mocked httpx; fast, no real server)."""

from __future__ import annotations

import httpx

from finsage.serving import health
from finsage.serving.health import check_openai_compatible_health, wait_for_vllm

MODELS_PAYLOAD = {"object": "list", "data": [{"id": "finsage-7b"}]}
BASE_URL = "http://localhost:8000/v1"


def test_health_returns_models(monkeypatch):
    """A healthy /models response yields healthy=True and the model list."""
    monkeypatch.setattr(
        health.httpx, "get", lambda url, timeout=5.0: httpx.Response(200, json=MODELS_PAYLOAD)
    )
    result = check_openai_compatible_health(BASE_URL)
    assert result["healthy"] is True
    assert result["models"] == ["finsage-7b"]


def test_health_handles_connection_error(monkeypatch):
    """A connection error yields healthy=False with an error message."""

    def boom(url, timeout=5.0):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(health.httpx, "get", boom)
    result = check_openai_compatible_health(BASE_URL)
    assert result["healthy"] is False
    assert result["status_code"] is None
    assert "refused" in result["error"]


def test_wait_for_vllm_succeeds(monkeypatch):
    """wait_for_vllm returns True once the endpoint reports healthy."""
    monkeypatch.setattr(health, "check_openai_compatible_health", lambda url: {"healthy": True})
    assert wait_for_vllm(BASE_URL, timeout_seconds=5, poll_interval_seconds=0) is True


def test_wait_for_vllm_times_out(monkeypatch):
    """wait_for_vllm returns False after the timeout with no healthy response."""
    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(health.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(health.time, "sleep", lambda _s: None)
    monkeypatch.setattr(health, "check_openai_compatible_health", lambda url: {"healthy": False})
    assert wait_for_vllm(BASE_URL, timeout_seconds=1, poll_interval_seconds=0) is False
