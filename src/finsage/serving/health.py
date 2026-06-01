"""Health-check helpers for the OpenAI-compatible vLLM server.

vLLM can take minutes to load 7B weights, so :func:`wait_for_vllm` polls the
``/models`` endpoint until it responds or a timeout elapses.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


def check_openai_compatible_health(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Check the ``/models`` endpoint of an OpenAI-compatible server.

    Args:
        base_url: Base URL (e.g. ``http://localhost:8000/v1``).
        timeout: Per-request timeout in seconds.

    Returns:
        A dict with ``healthy`` (bool), ``status_code`` (int | None), and either
        ``models`` (list of served model ids) or ``error``.
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        return {"healthy": False, "status_code": None, "error": str(exc)}

    if response.status_code != 200:
        return {"healthy": False, "status_code": response.status_code, "error": response.text[:200]}

    try:
        payload = response.json()
        models = [entry.get("id") for entry in payload.get("data", [])]
    except (ValueError, AttributeError) as exc:
        return {"healthy": False, "status_code": 200, "error": f"malformed payload: {exc}"}

    return {"healthy": True, "status_code": 200, "models": models}


def wait_for_vllm(
    base_url: str,
    timeout_seconds: int = 600,
    poll_interval_seconds: int = 5,
) -> bool:
    """Poll the vLLM ``/models`` endpoint until it is healthy or times out.

    Args:
        base_url: Base URL (e.g. ``http://localhost:8000/v1``).
        timeout_seconds: Maximum time to wait.
        poll_interval_seconds: Delay between polls.

    Returns:
        ``True`` if the server became healthy within the timeout, else ``False``.
    """
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        health = check_openai_compatible_health(base_url)
        if health.get("healthy"):
            logger.info("vLLM healthy after %d attempt(s): %s", attempt, health.get("models"))
            return True
        logger.info("vLLM not ready (attempt %d); retrying in %ss", attempt, poll_interval_seconds)
        time.sleep(poll_interval_seconds)
    logger.error("vLLM did not become healthy within %ds", timeout_seconds)
    return False
