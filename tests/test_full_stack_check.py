"""Tests for scripts/check_full_stack.py using a mocked HTTP client."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from scripts.check_full_stack import run_checks, write_report

MODELS_PAYLOAD = {"object": "list", "data": [{"id": "finsage-7b"}]}
CHAT_PAYLOAD = {
    "answer": "Key risks: supply chain and competition.",
    "model": "finsage-7b",
    "task_type": "risk_summary",
    "disclaimer": "Not financial advice.",
    "request_id": "req-1",
    "latency_ms": 120.0,
}


def _client(handler) -> httpx.Client:
    """Build an httpx.Client backed by a MockTransport handler."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _healthy_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.rstrip("/").endswith(":3000") or url.rstrip("/") == "http://localhost:3000":
        return httpx.Response(200, text="<!doctype html>")
    if url.endswith("/health"):
        return httpx.Response(200, json={"status": "ok", "service": "finsage-api"})
    if url.endswith("/ready"):
        return httpx.Response(200, json={"status": "ready", "vllm_available": True})
    if url.endswith("/chat"):
        return httpx.Response(200, json=CHAT_PAYLOAD)
    if url.endswith("/models"):
        return httpx.Response(200, json=MODELS_PAYLOAD)
    return httpx.Response(404, text="not found")


def test_all_checks_pass_when_healthy():
    """A fully healthy stack reports all_ok and individual successes."""
    report = run_checks(
        "http://localhost:3000",
        "http://localhost:8080/v1",
        "http://localhost:8000/v1",
        "change-me",
        client=_client(_healthy_handler),
    )
    assert report["all_ok"] is True
    names = {c["name"]: c["ok"] for c in report["checks"]}
    assert names["frontend reachable"] is True
    assert names["API /v1/health"] is True
    assert names["API /v1/chat"] is True
    assert names["vLLM /v1/models"] is True


def test_vllm_skipped_in_demo_mode():
    """Demo mode skips the vLLM check but still verifies frontend + API."""
    report = run_checks(
        "http://localhost:3000",
        "http://localhost:8080/v1",
        "http://localhost:8000/v1",
        "change-me",
        demo_mode=True,
        client=_client(_healthy_handler),
    )
    vllm = next(c for c in report["checks"] if c["name"] == "vLLM /v1/models")
    assert vllm["ok"] is True
    assert "skipped" in vllm["detail"]
    assert report["all_ok"] is True


def test_unavailable_service_fails_cleanly():
    """A connection error on the API is reported as a clean failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/health"):
            raise httpx.ConnectError("refused")
        return _healthy_handler(request)

    report = run_checks(
        "http://localhost:3000",
        "http://localhost:8080/v1",
        "http://localhost:8000/v1",
        "change-me",
        client=_client(handler),
    )
    health = next(c for c in report["checks"] if c["name"] == "API /v1/health")
    assert health["ok"] is False
    assert report["all_ok"] is False


def test_write_report_creates_json(tmp_path: Path):
    """write_report serialises the report to JSON at the given path."""
    report = run_checks(
        "http://localhost:3000",
        "http://localhost:8080/v1",
        "http://localhost:8000/v1",
        "change-me",
        demo_mode=True,
        client=_client(_healthy_handler),
    )
    out = tmp_path / "nested" / "full_stack_health.json"
    written = write_report(report, str(out))
    assert written.exists()
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["all_ok"] is True
    assert isinstance(loaded["checks"], list)
