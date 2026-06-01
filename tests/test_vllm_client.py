"""Tests for VLLMClient using httpx.MockTransport (no real server)."""

from __future__ import annotations

import httpx
import pytest

from finsage.serving.vllm_client import VLLMClient, VLLMClientError

MODELS_PAYLOAD = {"object": "list", "data": [{"id": "finsage-7b", "object": "model"}]}
CHAT_PAYLOAD = {
    "id": "chatcmpl-1",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Key risks: competition."}}
    ],
    "usage": {"completion_tokens": 5},
}


def _client(handler) -> VLLMClient:
    """Build a VLLMClient backed by a MockTransport handler."""
    return VLLMClient(transport=httpx.MockTransport(handler))


def test_health_calls_models():
    """health() hits /models and returns the payload."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=MODELS_PAYLOAD)

    data = _client(handler).health()
    assert seen["url"].endswith("/v1/models")
    assert data["data"][0]["id"] == "finsage-7b"


def test_chat_calls_chat_completions():
    """chat() hits /chat/completions and returns choices."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=CHAT_PAYLOAD)

    data = _client(handler).chat("Summarize the risks.")
    assert seen["url"].endswith("/v1/chat/completions")
    assert data["choices"]


def test_chat_text_extracts_content():
    """chat_text() returns just the assistant content."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_PAYLOAD)

    assert _client(handler).chat_text("Summarize.") == "Key risks: competition."


def test_authorization_header_included_with_api_key():
    """An Authorization bearer header is sent only when an api_key is set."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=MODELS_PAYLOAD)

    VLLMClient(api_key="secret", transport=httpx.MockTransport(handler)).health()
    assert seen["auth"] == "Bearer secret"

    VLLMClient(transport=httpx.MockTransport(handler)).health()
    assert seen["auth"] is None


def test_error_on_non_200():
    """A non-2xx response raises VLLMClientError with the status code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(VLLMClientError, match="500"):
        _client(handler).health()


def test_error_on_malformed_chat_response():
    """A chat response without 'choices' raises VLLMClientError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x"})  # no choices

    with pytest.raises(VLLMClientError, match="Malformed"):
        _client(handler).chat("hi")


def test_error_on_network_failure():
    """A transport/network error is wrapped as VLLMClientError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(VLLMClientError):
        _client(handler).health()
