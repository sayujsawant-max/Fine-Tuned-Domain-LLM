"""Prompt formatting for baseline evaluation.

Builds grounded, task-aware prompts (plain string and OpenAI-style chat
messages) and normalises raw model output into a clean prediction. No
model-specific behaviour beyond an optional chat-template path handled by the
generator; these helpers are model-agnostic.
"""

from __future__ import annotations

import re

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: System instruction shared by every evaluation prompt.
SYSTEM_INSTRUCTION = (
    "You are a financial filing analysis assistant. Answer using only the "
    "provided filing excerpt. If the answer is not supported by the excerpt, "
    "say so."
)

# Special tokens / chat markers stripped from raw generations.
_SPECIAL_TOKENS_RE = re.compile(
    r"</?s>|\[/?INST\]|<\|[^>]*\|>|<<SYS>>|<</SYS>>",
    re.IGNORECASE,
)
_DISCLAIMER_RE = re.compile(
    r"FinSage-7B is not a licensed financial advisor\.[^\n]*", re.IGNORECASE
)


def _user_content(example: dict) -> str:
    """Build the user-facing portion of an evaluation prompt.

    Args:
        example: An instruction example with ``task_type``, ``instruction``,
            and ``input`` fields.

    Returns:
        The user message content.
    """
    task_type = str(example.get("task_type", "general"))
    instruction = str(example.get("instruction", "")).strip()
    filing_input = str(example.get("input", "")).strip()
    return (
        f"Task type: {task_type}\n"
        f"Instruction: {instruction}\n\n"
        f"Filing excerpt:\n{filing_input}\n\n"
        "Provide a concise, filing-grounded answer."
    )


def build_eval_prompt(example: dict) -> str:
    """Build a single plain-text evaluation prompt.

    Args:
        example: An instruction example.

    Returns:
        A prompt string combining the system instruction, task type,
        instruction, and filing excerpt.
    """
    return f"{SYSTEM_INSTRUCTION}\n\n{_user_content(example)}"


def build_chat_messages(example: dict) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for an example.

    Args:
        example: An instruction example.

    Returns:
        A two-message list: a system message and a user message.
    """
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": _user_content(example)},
    ]


def normalize_prediction(text: str) -> str:
    """Clean a raw model generation into a comparable prediction.

    Removes chat/special tokens, injected disclaimers, repeated blank lines, and
    surrounding whitespace.

    Args:
        text: The raw generated text.

    Returns:
        The normalised prediction string.
    """
    if not text:
        return ""
    cleaned = _SPECIAL_TOKENS_RE.sub(" ", text)
    cleaned = _DISCLAIMER_RE.sub(" ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty).strip()
