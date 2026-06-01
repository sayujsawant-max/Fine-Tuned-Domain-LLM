"""Tests for the FastAPI service using the in-process TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from finsage.serving.app import app
from finsage.serving.schemas import DISCLAIMER

client = TestClient(app)


def test_health_endpoint():
    """GET /v1/health returns 200 with status, version, and model."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]
    assert body["version"]


def test_chat_endpoint_returns_answer_and_disclaimer():
    """POST /v1/chat returns a mock answer carrying the disclaimer."""
    response = client.post(
        "/v1/chat",
        json={
            "question": "What are the top three risk factors?",
            "context": "Item 1A. Risk Factors ...",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["disclaimer"] == DISCLAIMER
    assert body["grounded"] is True


def test_chat_without_context_is_not_grounded():
    """A request without context is reported as not grounded."""
    response = client.post("/v1/chat", json={"question": "Hello?"})
    assert response.status_code == 200
    assert response.json()["grounded"] is False


def test_chat_requires_question():
    """An empty question fails validation with HTTP 422."""
    response = client.post("/v1/chat", json={"question": ""})
    assert response.status_code == 422
