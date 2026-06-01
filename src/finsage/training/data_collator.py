"""Data collation for instruction fine-tuning (Phase 6 stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: Prompt template used to render an instruction example into model input text.
PROMPT_TEMPLATE = "<s>[INST] {instruction}\n\n{input} [/INST] {output}</s>"


def format_example(example: dict[str, Any]) -> str:
    """Render an instruction example into a single training string.

    Args:
        example: A dict with ``instruction``, ``input``, and ``output`` keys.

    Returns:
        The formatted prompt string following the Mistral instruct template.
    """
    return PROMPT_TEMPLATE.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", ""),
    )


@dataclass
class InstructionDataCollator:
    """Collate formatted instruction examples into model-ready batches.

    Args:
        max_length: Maximum sequence length; longer examples are truncated.
    """

    max_length: int = 2048

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of examples.

        Args:
            features: A batch of instruction example dicts.

        Returns:
            A dict of tensors ready for the model.

        Raises:
            NotImplementedError: Always, until Phase 6 (requires a tokenizer
                from the ``training`` extras).
        """
        raise NotImplementedError(
            "Tokenized collation lands in Phase 6 (install the 'training' extras)."
        )
