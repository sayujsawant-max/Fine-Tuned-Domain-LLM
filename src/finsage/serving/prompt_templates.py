"""Prompt templating for FinSage-7B chat requests.

Builds OpenAI-style ``messages`` that ground the model in a supplied filing
excerpt and forbid investment advice. The system prompt is the contract between
the API and the model.
"""

from __future__ import annotations

from finsage.config import SUPPORTED_TASK_TYPES

#: Grounding system prompt prepended to every chat request.
SYSTEM_PROMPT = (
    "You are FinSage-7B, a financial filing analysis assistant. Answer using "
    "only the provided filing excerpt. Do not provide investment advice. If the "
    "answer is not supported by the excerpt, say so."
)


def build_system_prompt() -> str:
    """Return the grounding system prompt.

    Returns:
        The :data:`SYSTEM_PROMPT` string.
    """
    return SYSTEM_PROMPT


def validate_task_type(task_type: str | None) -> str | None:
    """Validate an optional task-type hint.

    Args:
        task_type: The task type to validate, or ``None``.

    Returns:
        The validated task type, or ``None`` when not provided.

    Raises:
        ValueError: If ``task_type`` is not a supported task type.
    """
    if task_type is None:
        return None
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type {task_type!r}; expected one of {sorted(SUPPORTED_TASK_TYPES)}"
        )
    return task_type


def build_filing_chat_prompt(
    question: str,
    filing_excerpt: str,
    task_type: str | None = None,
) -> list[dict[str, str]]:
    """Build OpenAI-style messages for a grounded filing question.

    Args:
        question: The user's question about the filing.
        filing_excerpt: The filing text the answer must be grounded in.
        task_type: Optional task-type hint (validated against the supported set).

    Returns:
        A two-element list of ``{"role", "content"}`` messages: a system message
        with the grounding instruction and a user message embedding the task
        type, question, and filing excerpt.

    Raises:
        ValueError: If ``task_type`` is provided but unsupported.
    """
    task = validate_task_type(task_type)
    task_line = f"Task Type: {task}\n" if task else ""
    user_content = f"{task_line}" f"Question: {question}\n" f"Filing Excerpt:\n{filing_excerpt}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
