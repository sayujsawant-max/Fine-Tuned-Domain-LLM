"""Format instruction examples into supervised fine-tuning (SFT) text.

Pure-Python and dependency-free so dry-run validation and tests work on a plain
CPU machine without transformers/torch installed. The prompt embeds a disclaimer
instruction and never provides investment advice.
"""

from __future__ import annotations

from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: Required fields for a valid training example.
REQUIRED_FIELDS = ("instruction", "input", "output", "task_type")

#: SFT text template (Mistral instruct format).
SFT_TEMPLATE = (
    "<s>[INST] You are FinSage-7B, a financial filing analysis assistant.\n"
    "Answer using only the provided filing excerpt. Do not provide investment advice.\n\n"
    "Task Type:\n{task_type}\n\n"
    "Instruction:\n{instruction}\n\n"
    "Filing Excerpt:\n{input}\n"
    "[/INST]\n{output}</s>"
)


def format_sft_example(example: dict[str, Any]) -> str:
    """Render one instruction example into a single SFT training string.

    Args:
        example: An instruction example with ``task_type``, ``instruction``,
            ``input``, and ``output`` fields.

    Returns:
        The formatted training text following the Mistral instruct template.
    """
    return SFT_TEMPLATE.format(
        task_type=str(example.get("task_type", "")).strip(),
        instruction=str(example.get("instruction", "")).strip(),
        input=str(example.get("input", "")).strip(),
        output=str(example.get("output", "")).strip(),
    )


def format_dataset_for_sft(dataset: Any) -> Any:
    """Add a ``text`` field with the formatted SFT string to each example.

    Works with a Hugging Face ``Dataset`` (uses ``.map``) or a plain list of
    example dicts.

    Args:
        dataset: A Hugging Face dataset or a list of example dicts.

    Returns:
        The dataset with a ``text`` field added to every example (same type as
        the input).
    """
    if hasattr(dataset, "map"):
        return dataset.map(lambda example: {"text": format_sft_example(example)})
    return [{**example, "text": format_sft_example(example)} for example in dataset]


def validate_training_example(example: dict[str, Any]) -> list[str]:
    """Validate a single training example.

    Args:
        example: The example to validate.

    Returns:
        A list of human-readable error strings; empty if the example is valid.
    """
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in example:
            errors.append(f"missing field '{field}'")
        elif example[field] is None or str(example[field]).strip() == "":
            errors.append(f"empty field '{field}'")
    return errors


def count_tokens_approx(text: str) -> int:
    """Approximate a token count using whitespace splitting (for logging only).

    Args:
        text: The text to measure.

    Returns:
        The number of whitespace-delimited tokens.
    """
    return len(text.split())
