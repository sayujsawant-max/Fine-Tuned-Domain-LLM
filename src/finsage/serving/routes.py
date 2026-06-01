"""API route definitions for the FinSage-7B service.

Routes proxy to the internal vLLM OpenAI-compatible server via an injected
:class:`VLLMClient`, layering on auth, disclaimer injection, and a clean
request/response contract. The client is provided through
:func:`get_vllm_client` so it can be mocked in tests.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from finsage import __version__
from finsage.config import SUPPORTED_TASK_TYPES, Settings, get_settings
from finsage.logging_utils import get_logger
from finsage.serving.auth import require_api_key
from finsage.serving.disclaimer import append_disclaimer, get_disclaimer
from finsage.serving.errors import VLLMUnavailableError
from finsage.serving.prompt_templates import build_filing_chat_prompt
from finsage.serving.schemas import (
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    HealthResponse,
    OpenAIChatCompletionRequest,
    ReadinessResponse,
)
from finsage.serving.vllm_client import VLLMClient, VLLMClientError, extract_message_content

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["finsage"])


def get_vllm_client() -> VLLMClient:
    """Build a :class:`VLLMClient` from settings (FastAPI dependency).

    Override this dependency in tests to inject a mock client.

    Returns:
        A configured :class:`VLLMClient` pointed at the internal vLLM server.
    """
    settings = get_settings()
    return VLLMClient(
        base_url=settings.effective_vllm_base_url,
        model=settings.served_model_name,
        timeout=settings.request_timeout_seconds,
    )


def _request_id(request: Request) -> str:
    """Return the request id set by the middleware (empty string if absent)."""
    return getattr(request.state, "request_id", "") or ""


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return liveness information. Does not require auth or vLLM.

    Returns:
        A :class:`HealthResponse` with status, service name, and version.
    """
    return HealthResponse(status="ok", service="finsage-api", version=__version__)


@router.get("/ready", response_model=ReadinessResponse, dependencies=[Depends(require_api_key)])
async def ready(
    client: VLLMClient = Depends(get_vllm_client),
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Report whether the API can reach the vLLM backend.

    Args:
        client: The vLLM client dependency.
        settings: The application settings dependency.

    Returns:
        A :class:`ReadinessResponse` describing backend availability.
    """
    try:
        await client.async_health()
    except VLLMClientError as exc:
        logger.warning("Readiness check failed: %s", exc)
        return ReadinessResponse(
            status="not_ready", vllm_available=False, model=None, error=str(exc)
        )
    return ReadinessResponse(status="ready", vllm_available=True, model=settings.served_model_name)


@router.get("/models", dependencies=[Depends(require_api_key)])
async def models(
    client: VLLMClient = Depends(get_vllm_client),
) -> dict[str, object]:
    """Proxy the vLLM model list.

    Args:
        client: The vLLM client dependency.

    Returns:
        The vLLM ``/models`` payload.

    Raises:
        VLLMUnavailableError: If the backend cannot be reached.
    """
    try:
        return await client.async_health()
    except VLLMClientError as exc:
        raise VLLMUnavailableError(f"vLLM backend unavailable: {exc}") from exc


@router.get("/config", response_model=ConfigResponse, dependencies=[Depends(require_api_key)])
async def config(settings: Settings = Depends(get_settings)) -> ConfigResponse:
    """Return safe, public-facing configuration (never secrets).

    Args:
        settings: The application settings dependency.

    Returns:
        A :class:`ConfigResponse` with model name, limits, and task types.
    """
    return ConfigResponse(
        model=settings.served_model_name,
        max_tokens_limit=2048,
        supported_task_types=list(SUPPORTED_TASK_TYPES),
        disclaimer_enabled=settings.disclaimer_enabled,
        rate_limit_requests_per_minute=settings.rate_limit_requests_per_minute,
    )


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(
    payload: ChatRequest,
    request: Request,
    client: VLLMClient = Depends(get_vllm_client),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """Answer a grounded question about a filing excerpt.

    Args:
        payload: The validated chat request.
        request: The incoming request (for the request id).
        client: The vLLM client dependency.
        settings: The application settings dependency.

    Returns:
        A :class:`ChatResponse` with the answer, model, disclaimer, request id,
        and latency.

    Raises:
        VLLMUnavailableError: If the backend cannot be reached.
    """
    start = time.perf_counter()
    messages = build_filing_chat_prompt(
        question=payload.question,
        filing_excerpt=payload.filing_excerpt,
        task_type=payload.task_type,
    )
    try:
        data = await client.async_chat(
            messages=messages,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
        answer = extract_message_content(data)
    except VLLMClientError as exc:
        raise VLLMUnavailableError(f"vLLM backend unavailable: {exc}") from exc

    disclaimer: str | None = None
    if settings.disclaimer_enabled and payload.include_disclaimer:
        answer = append_disclaimer(answer)
        disclaimer = get_disclaimer()

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ChatResponse(
        answer=answer,
        model=settings.served_model_name,
        task_type=payload.task_type,
        disclaimer=disclaimer,
        request_id=_request_id(request),
        latency_ms=latency_ms,
    )


@router.post("/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(
    payload: OpenAIChatCompletionRequest,
    client: VLLMClient = Depends(get_vllm_client),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """OpenAI-compatible chat-completions proxy.

    Forwards the request to vLLM, applying auth/logging/rate limiting upstream
    and optionally appending the disclaimer to the assistant message.

    Args:
        payload: The OpenAI-style request.
        client: The vLLM client dependency.
        settings: The application settings dependency.

    Returns:
        The vLLM chat-completion response (OpenAI shape preserved).

    Raises:
        VLLMUnavailableError: If the backend cannot be reached.
        FinsageAPIError: With HTTP 400 if streaming is requested (unsupported).
    """
    if payload.stream:
        from finsage.serving.errors import FinsageAPIError

        error = FinsageAPIError("Streaming is not supported in Phase 8.")
        error.status_code = 400
        error.error = "streaming_unsupported"
        raise error

    body = {
        "model": payload.model or settings.served_model_name,
        "messages": [m.model_dump() for m in payload.messages],
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
    }
    try:
        data = await client.async_chat_completion(body)
    except VLLMClientError as exc:
        raise VLLMUnavailableError(f"vLLM backend unavailable: {exc}") from exc

    if settings.disclaimer_enabled:
        try:
            message = data["choices"][0]["message"]  # type: ignore[index]
            message["content"] = append_disclaimer(str(message.get("content", "")))
        except (KeyError, IndexError, TypeError):
            logger.warning("Could not append disclaimer to chat/completions response")
    return data
