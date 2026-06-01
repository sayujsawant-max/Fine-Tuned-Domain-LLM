"""Route tests for the FinSage-7B API using FastAPI's TestClient.

All vLLM interaction is mocked via the ``make_api_app`` fixture's
``FakeVLLMClient`` — no real server, GPU, or weights are required.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from finsage.serving.disclaimer import FINANCIAL_DISCLAIMER
from tests.conftest import FakeVLLMClient

AUTH = {"X-API-Key": "test-secret"}
CHAT_BODY = {
    "question": "Summarize the main risk factors.",
    "filing_excerpt": "The company faces supply chain disruption and competition.",
    "task_type": "risk_summary",
}


def test_health_no_auth(make_api_app):
    """GET /v1/health works without auth."""
    client = TestClient(make_api_app())
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "service": "finsage-api", "version": body["version"]}


def test_ready_requires_auth(make_api_app):
    """GET /v1/ready returns 401 without a key."""
    client = TestClient(make_api_app())
    assert client.get("/v1/ready").status_code == 401


def test_ready_when_vllm_available(make_api_app):
    """GET /v1/ready reports ready when the backend responds."""
    client = TestClient(make_api_app())
    resp = client.get("/v1/ready", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["vllm_available"] is True
    assert body["model"] == "finsage-7b"


def test_ready_when_vllm_down(make_api_app):
    """GET /v1/ready reports not_ready when the backend is down."""
    app = make_api_app(vllm=FakeVLLMClient(available=False))
    resp = TestClient(app).get("/v1/ready", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["vllm_available"] is False
    assert body["error"]


def test_models_requires_auth(make_api_app):
    """GET /v1/models requires auth and proxies the model list."""
    app = make_api_app()
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 401
    resp = client.get("/v1/models", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "finsage-7b"


def test_config_does_not_expose_secret(make_api_app):
    """GET /v1/config returns safe config without secrets."""
    resp = TestClient(make_api_app()).get("/v1/config", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert "test-secret" not in resp.text
    assert "api_secret_key" not in body
    assert body["model"] == "finsage-7b"
    assert "risk_summary" in body["supported_task_types"]
    assert body["max_tokens_limit"] == 2048


def test_chat_requires_auth(make_api_app):
    """POST /v1/chat returns 401 without a key."""
    resp = TestClient(make_api_app()).post("/v1/chat", json=CHAT_BODY)
    assert resp.status_code == 401


def test_chat_invalid_key_fails(make_api_app):
    """POST /v1/chat with a wrong key returns 401."""
    resp = TestClient(make_api_app()).post(
        "/v1/chat", json=CHAT_BODY, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401


def test_chat_returns_answer_and_disclaimer(make_api_app):
    """POST /v1/chat returns the answer plus the disclaimer."""
    resp = TestClient(make_api_app()).post("/v1/chat", json=CHAT_BODY, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert "supply chain disruption" in body["answer"]
    assert body["disclaimer"] == FINANCIAL_DISCLAIMER
    assert FINANCIAL_DISCLAIMER in body["answer"]
    assert body["task_type"] == "risk_summary"
    assert body["model"] == "finsage-7b"
    assert body["request_id"]
    assert body["latency_ms"] >= 0


def test_chat_without_disclaimer(make_api_app):
    """include_disclaimer=false suppresses disclaimer injection."""
    body = {**CHAT_BODY, "include_disclaimer": False}
    resp = TestClient(make_api_app()).post("/v1/chat", json=body, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["disclaimer"] is None
    assert FINANCIAL_DISCLAIMER not in data["answer"]


def test_chat_validation_error(make_api_app):
    """An empty question returns 422 with the uniform error shape."""
    resp = TestClient(make_api_app()).post(
        "/v1/chat", json={"question": "", "filing_excerpt": "x"}, headers=AUTH
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error"
    assert "request_id" in body


def test_chat_vllm_unavailable_returns_503(make_api_app):
    """A backend failure surfaces as HTTP 503."""
    app = make_api_app(vllm=FakeVLLMClient(available=False))
    resp = TestClient(app).post("/v1/chat", json=CHAT_BODY, headers=AUTH)
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "vllm_unavailable"
    assert body["request_id"]


def test_chat_completions_rejects_stream(make_api_app):
    """POST /v1/chat/completions with stream=true returns 400."""
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = TestClient(make_api_app()).post("/v1/chat/completions", json=body, headers=AUTH)
    assert resp.status_code == 400
    assert "Streaming is not supported" in resp.json()["detail"]


def test_chat_completions_proxies_and_appends_disclaimer(make_api_app):
    """POST /v1/chat/completions preserves OpenAI shape and adds disclaimer."""
    body = {"messages": [{"role": "user", "content": "Summarize risks."}]}
    resp = TestClient(make_api_app()).post("/v1/chat/completions", json=body, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["usage"]["total_tokens"] == 54
    assert FINANCIAL_DISCLAIMER in data["choices"][0]["message"]["content"]


def test_rate_limit_returns_429(make_api_app):
    """Exceeding the per-client budget returns 429 with headers."""
    client = TestClient(make_api_app(rate_limit=1))
    first = client.get("/v1/config", headers=AUTH)
    assert first.status_code == 200
    assert first.headers["X-RateLimit-Limit"] == "1"
    second = client.get("/v1/config", headers=AUTH)
    assert second.status_code == 429
    assert second.json()["error"] == "rate_limit_exceeded"
    assert second.headers["X-RateLimit-Remaining"] == "0"


def test_request_id_header_present(make_api_app):
    """Every response carries an X-Request-ID header."""
    resp = TestClient(make_api_app()).get("/v1/health")
    assert resp.headers["X-Request-ID"]


def test_security_headers_present(make_api_app):
    """Security headers are set on responses."""
    resp = TestClient(make_api_app()).get("/v1/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["X-XSS-Protection"] == "0"
    assert resp.headers["Referrer-Policy"] == "no-referrer"


def test_dev_mode_allows_without_key(make_api_app):
    """With a placeholder secret in dev mode, requests are allowed."""
    app = make_api_app(secret="change-me", environment="development")
    resp = TestClient(app).get("/v1/config")
    assert resp.status_code == 200


def test_production_rejects_placeholder_secret(make_api_app):
    """With a placeholder secret in production, requests are rejected."""
    app = make_api_app(secret="change-me", environment="production")
    resp = TestClient(app).get("/v1/config", headers={"X-API-Key": "change-me"})
    assert resp.status_code == 401
