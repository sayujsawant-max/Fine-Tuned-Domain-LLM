"""Pydantic v2 request/response models for the FinSage-7B API."""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Mandatory disclaimer appended to every model response.
DISCLAIMER = (
    "FinSage-7B is not a licensed financial advisor. Outputs are not investment "
    "recommendations. Verify all responses against the original filings."
)


class ChatRequest(BaseModel):
    """A request to analyse a filing excerpt.

    Attributes:
        question: The user's question about the filing.
        context: Optional filing excerpt the answer should be grounded in.
        task_type: Optional task hint (e.g. ``"risk_summary"``, ``"filing_qa"``).
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature; ``0`` for deterministic output.
    """

    question: str = Field(..., min_length=1, description="Question about the filing.")
    context: str | None = Field(
        default=None, description="Optional filing excerpt to ground the answer."
    )
    task_type: str | None = Field(default=None, description="Optional task hint.")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """A model response with provenance and disclaimer.

    Attributes:
        answer: The generated answer text.
        model: The model name that produced the answer.
        disclaimer: The mandatory financial disclaimer.
        grounded: Whether the answer was produced against supplied context.
    """

    answer: str
    model: str
    disclaimer: str = DISCLAIMER
    grounded: bool = False


class HealthResponse(BaseModel):
    """Service health payload.

    Attributes:
        status: Health status string (``"ok"`` when serving).
        version: Application version.
        model: Configured model identifier.
    """

    status: str = "ok"
    version: str
    model: str
