"""Tests for the API middleware stack (request id, logging, security)."""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

SECRET_EXCERPT = "SECRET-FILING-TEXT-supply-chain-disruption-12345"
AUTH = {"X-API-Key": "test-secret"}


def test_request_id_header_exists(make_api_app):
    """The X-Request-ID header is present on responses."""
    resp = TestClient(make_api_app()).get("/v1/health")
    assert resp.headers.get("X-Request-ID")


def test_security_headers_exist(make_api_app):
    """All baseline security headers are present."""
    resp = TestClient(make_api_app()).get("/v1/health")
    for header in (
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection",
        "Referrer-Policy",
    ):
        assert header in resp.headers


def test_logging_excludes_filing_text(make_api_app, caplog):
    """Structured logs never contain the filing excerpt content."""
    client = TestClient(make_api_app())
    with caplog.at_level(logging.INFO, logger="finsage.serving.middleware"):
        resp = client.post(
            "/v1/chat",
            json={"question": "Summarize.", "filing_excerpt": SECRET_EXCERPT},
            headers=AUTH,
        )
    assert resp.status_code == 200
    for record in caplog.records:
        assert SECRET_EXCERPT not in record.getMessage()


def test_logging_records_latency(make_api_app, caplog):
    """The structured log line records latency and a request id."""
    client = TestClient(make_api_app())
    with caplog.at_level(logging.INFO, logger="finsage.serving.middleware"):
        client.get("/v1/health")
    json_records = []
    for record in caplog.records:
        try:
            json_records.append(json.loads(record.getMessage()))
        except (ValueError, TypeError):
            continue
    assert json_records, "expected at least one JSON log line"
    latest = json_records[-1]
    assert "latency_ms" in latest
    assert latest["latency_ms"] >= 0
    assert latest["request_id"]
    assert latest["path"] == "/v1/health"
