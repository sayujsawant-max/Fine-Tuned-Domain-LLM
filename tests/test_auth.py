"""Tests for API key authentication helpers."""

from __future__ import annotations

from finsage.serving.auth import (
    get_api_key_from_headers,
    is_public_path,
    verify_api_key,
)


def test_get_key_from_x_api_key_header():
    """X-API-Key is read (case-insensitively)."""
    assert get_api_key_from_headers({"x-api-key": "abc"}) == "abc"


def test_get_key_from_bearer_token():
    """Authorization: Bearer <key> is read."""
    assert get_api_key_from_headers({"authorization": "Bearer xyz"}) == "xyz"


def test_x_api_key_takes_precedence():
    """X-API-Key wins when both headers are present."""
    headers = {"x-api-key": "primary", "authorization": "Bearer secondary"}
    assert get_api_key_from_headers(headers) == "primary"


def test_missing_key_returns_none():
    """No key header yields None."""
    assert get_api_key_from_headers({}) is None
    assert get_api_key_from_headers({"authorization": "Basic abc"}) is None


def test_verify_valid_key_passes():
    """Matching keys verify true."""
    assert verify_api_key("secret", "secret") is True


def test_verify_invalid_key_fails():
    """Mismatched keys verify false."""
    assert verify_api_key("wrong", "secret") is False


def test_verify_missing_key_fails():
    """Missing provided or expected key verifies false."""
    assert verify_api_key(None, "secret") is False
    assert verify_api_key("secret", None) is False
    assert verify_api_key(None, None) is False


def test_public_path_detection():
    """Health and docs paths are public; API paths are protected."""
    assert is_public_path("/v1/health") is True
    assert is_public_path("/docs") is True
    assert is_public_path("/openapi.json") is True
    assert is_public_path("/redoc") is True
    assert is_public_path("/v1/chat") is False
    assert is_public_path("/v1/ready") is False
    assert is_public_path("/v1/models") is False
    assert is_public_path("/v1/config") is False
