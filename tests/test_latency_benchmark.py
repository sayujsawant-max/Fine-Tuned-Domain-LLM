"""Tests for the latency benchmark against a mocked endpoint (fast, no server)."""

from __future__ import annotations

import httpx
from serving.benchmark_latency import run_benchmark

CHAT_PAYLOAD = {
    "choices": [{"message": {"role": "assistant", "content": "Key risks: competition."}}],
    "usage": {"completion_tokens": 12},
}


def test_benchmark_against_mocked_endpoint():
    """All requests succeed and percentile fields are present."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_PAYLOAD)

    metrics = run_benchmark(
        "http://localhost:8000/v1",
        "finsage-7b",
        num_requests=8,
        concurrency=2,
        transport=httpx.MockTransport(handler),
    )
    assert metrics["num_requests"] == 8
    assert metrics["successful_requests"] == 8
    assert metrics["failed_requests"] == 0
    for key in ("p50_latency_s", "p95_latency_s", "p99_latency_s"):
        assert key in metrics
    assert metrics["total_completion_tokens"] == 8 * 12


def test_benchmark_counts_failures():
    """Non-200 responses are counted as failures."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    metrics = run_benchmark(
        "http://localhost:8000/v1",
        "finsage-7b",
        num_requests=5,
        concurrency=1,
        transport=httpx.MockTransport(handler),
    )
    assert metrics["failed_requests"] == 5
    assert metrics["successful_requests"] == 0


def test_benchmark_output_has_no_nan_or_inf():
    """The metrics dict is JSON-serialisable with finite numbers only."""
    import json
    import math

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_PAYLOAD)

    metrics = run_benchmark(
        "http://localhost:8000/v1",
        "finsage-7b",
        num_requests=4,
        concurrency=1,
        transport=httpx.MockTransport(handler),
    )
    raw = json.dumps(metrics)
    assert "NaN" not in raw and "Infinity" not in raw
    for value in metrics.values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_benchmark_handles_all_failures_gracefully():
    """A fully-down endpoint yields zeroed latency metrics, not an error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    metrics = run_benchmark(
        "http://localhost:8000/v1",
        "finsage-7b",
        num_requests=3,
        concurrency=1,
        transport=httpx.MockTransport(handler),
    )
    assert metrics["successful_requests"] == 0
    assert metrics["p50_latency_s"] == 0.0
    assert metrics["avg_latency_s"] == 0.0
