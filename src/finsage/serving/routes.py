"""API route definitions for the FinSage-7B service.

Phase 1 returns a deterministic mock answer (no model is loaded) while still
exercising the full request/response contract, disclaimer injection, and
grounding flag. Real inference is delegated to the vLLM backend in Phase 8/9.
"""

from __future__ import annotations

from fastapi import APIRouter

from finsage import __version__
from finsage.config import get_settings
from finsage.logging_utils import get_logger
from finsage.serving.schemas import ChatRequest, ChatResponse, HealthResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["finsage"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and configuration metadata.

    Returns:
        A :class:`HealthResponse` with status, version, and configured model.
    """
    settings = get_settings()
    return HealthResponse(status="ok", version=__version__, model=settings.model_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question about a filing excerpt.

    In Phase 1 this returns a mock answer with the mandatory disclaimer; it does
    not load or call any model.

    Args:
        request: The chat request payload.

    Returns:
        A :class:`ChatResponse` with the (mock) answer, model name, disclaimer,
        and whether context was supplied.
    """
    settings = get_settings()
    grounded = bool(request.context and request.context.strip())
    logger.info("chat request: task_type=%s grounded=%s", request.task_type, grounded)
    answer = (
        "[Phase 1 mock response] FinSage-7B is not yet wired to a model. "
        f"Received question: {request.question!r}."
    )
    return ChatResponse(answer=answer, model=settings.model_id, grounded=grounded)
