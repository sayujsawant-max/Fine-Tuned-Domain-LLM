"""Serving layer for FinSage-7B: vLLM client/health helpers and the FastAPI app."""

from __future__ import annotations

from finsage.serving.health import check_openai_compatible_health, wait_for_vllm
from finsage.serving.vllm_client import VLLMClient, VLLMClientError, extract_message_content

__all__ = [
    "VLLMClient",
    "VLLMClientError",
    "check_openai_compatible_health",
    "create_app",
    "extract_message_content",
    "wait_for_vllm",
]


def __getattr__(name: str) -> object:
    """Lazily expose the app factory to avoid importing FastAPI eagerly."""
    if name == "create_app":
        from finsage.serving.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
