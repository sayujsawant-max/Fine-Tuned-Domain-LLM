"""Pydantic v2 request/response models for the FinSage-7B API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from finsage.config import SUPPORTED_TASK_TYPES
from finsage.serving.disclaimer import FINANCIAL_DISCLAIMER

#: Backwards-compatible alias for the canonical disclaimer string.
DISCLAIMER = FINANCIAL_DISCLAIMER


class HealthResponse(BaseModel):
    """Liveness payload for the FastAPI service.

    Attributes:
        status: Health status string (``"ok"`` when serving).
        service: Service identifier.
        version: Application version.
    """

    status: str = "ok"
    service: str = "finsage-api"
    version: str


class ReadinessResponse(BaseModel):
    """Readiness payload reporting whether the vLLM backend is reachable.

    Attributes:
        status: ``"ready"`` or ``"not_ready"``.
        vllm_available: Whether the vLLM backend responded.
        model: Served model name when known, otherwise ``None``.
        error: Error detail when the backend is unreachable.
    """

    status: str
    vllm_available: bool
    model: str | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    """A request to analyse a filing excerpt.

    Attributes:
        question: The user's question about the filing.
        filing_excerpt: Filing text the answer must be grounded in.
        task_type: Optional task hint; must be a supported task type.
        max_tokens: Maximum number of tokens to generate (1–2048).
        temperature: Sampling temperature (0.0–2.0); ``0`` is deterministic.
        include_disclaimer: Whether to append the financial disclaimer.
    """

    question: str = Field(..., min_length=1, description="Question about the filing.")
    filing_excerpt: str = Field(
        ..., min_length=1, description="Filing excerpt to ground the answer in."
    )
    task_type: str | None = Field(default=None, description="Optional task hint.")
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    include_disclaimer: bool = Field(default=True)

    @field_validator("question", "filing_excerpt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject whitespace-only ``question`` / ``filing_excerpt`` values."""
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("task_type")
    @classmethod
    def _known_task_type(cls, value: str | None) -> str | None:
        """Reject unsupported task types."""
        if value is None:
            return None
        if value not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {sorted(SUPPORTED_TASK_TYPES)}; got {value!r}"
            )
        return value


class ChatResponse(BaseModel):
    """A model response with provenance and disclaimer.

    Attributes:
        answer: The generated answer text (with disclaimer if requested).
        model: The served model name that produced the answer.
        task_type: The task type used, if any.
        disclaimer: The financial disclaimer, or ``None`` if disabled.
        request_id: Correlation ID for the request.
        latency_ms: End-to-end handler latency in milliseconds.
    """

    answer: str
    model: str
    task_type: str | None = None
    disclaimer: str | None = None
    request_id: str
    latency_ms: float


class ErrorResponse(BaseModel):
    """Uniform error payload returned by the exception handlers.

    Attributes:
        error: Short error category/message.
        detail: Optional human-readable detail.
        request_id: Correlation ID for the request, if available.
    """

    error: str
    detail: str | None = None
    request_id: str | None = None


class OpenAIChatMessage(BaseModel):
    """A single OpenAI-style chat message.

    Attributes:
        role: Message role (``"system"``, ``"user"``, ``"assistant"``).
        content: Message content.
    """

    role: str
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    """An OpenAI-compatible chat-completion request.

    Attributes:
        model: Optional model override; defaults to the served model.
        messages: Conversation history; must be non-empty.
        temperature: Sampling temperature (0.0–2.0).
        max_tokens: Maximum number of tokens to generate (1–2048).
        stream: Streaming flag; unsupported in Phase 8 (rejected with 400).
    """

    model: str | None = None
    messages: list[OpenAIChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    stream: bool = False


class ConfigResponse(BaseModel):
    """Safe, public-facing API configuration.

    Attributes:
        model: Served model name.
        max_tokens_limit: Upper bound on ``max_tokens``.
        supported_task_types: Task types accepted by ``/chat``.
        disclaimer_enabled: Whether the disclaimer is injected.
        rate_limit_requests_per_minute: Per-client request budget per minute.
    """

    model: str
    max_tokens_limit: int
    supported_task_types: list[str]
    disclaimer_enabled: bool
    rate_limit_requests_per_minute: int
