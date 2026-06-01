"""Turn filing chunks into JSONL-ready instruction-tuning examples.

Phase 1 wires up the ten task types and the example schema. The ``output`` field
is left as an empty placeholder because real targets are generated in Phase 4
(by a stronger teacher model and/or extracted from the filing), and must never
be fabricated here.
"""

from __future__ import annotations

from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: The ten supported instruction task types, in blueprint order.
TASK_TYPES: tuple[str, ...] = (
    "risk_summary",
    "mda_explanation",
    "metric_extraction",
    "yoy_comparison",
    "business_risk_identification",
    "revenue_driver_explanation",
    "filing_qa",
    "analyst_summary",
    "outlook_classification",
    "hallucination_detection",
)

#: Human-readable instruction prompt for each task type.
_INSTRUCTION_TEMPLATES: dict[str, str] = {
    "risk_summary": "Summarize the top three risk factors disclosed in this filing excerpt.",
    "mda_explanation": "Explain what management discusses in this MD&A excerpt in plain language.",
    "metric_extraction": "Extract the reported financial metrics and their values from this excerpt.",
    "yoy_comparison": "Compare the reported figures year over year based on this excerpt.",
    "business_risk_identification": "Identify the key business risks described in this excerpt.",
    "revenue_driver_explanation": "Explain the main revenue drivers described in this excerpt.",
    "filing_qa": "Answer the question using only the filing excerpt provided.",
    "analyst_summary": "Write a concise analyst-style summary of this filing excerpt.",
    "outlook_classification": (
        "Classify the financial outlook expressed in this excerpt as "
        "positive, neutral, or negative."
    ),
    "hallucination_detection": (
        "Determine whether the claim is supported by the filing excerpt. "
        "Answer 'supported' or 'unsupported' and justify briefly."
    ),
}


def _make_example_id(metadata: dict[str, Any], task_type: str, chunk_id: Any) -> str:
    """Build a stable, human-readable example id.

    Args:
        metadata: Chunk metadata (expects ``ticker``, ``year``, ``filing_type``).
        task_type: The task type for this example.
        chunk_id: The originating chunk's identifier.

    Returns:
        An id such as ``AAPL-2022-10-K-RISK_SUMMARY-0042``.
    """
    ticker = str(metadata.get("ticker", "UNK")).upper()
    year = metadata.get("year", "XXXX")
    filing_type = str(metadata.get("filing_type", "NA")).upper()
    try:
        suffix = f"{int(chunk_id):04d}"
    except (TypeError, ValueError):
        suffix = str(chunk_id)
    return f"{ticker}-{year}-{filing_type}-{task_type.upper()}-{suffix}"


class InstructionBuilder:
    """Builds instruction-tuning examples from filing chunks."""

    task_types: tuple[str, ...] = TASK_TYPES

    def build_example(
        self,
        chunk: dict[str, Any],
        task_type: str,
        question: str | None = None,
        output: str = "",
    ) -> dict[str, Any]:
        """Build a single JSONL-ready instruction example.

        Args:
            chunk: A chunk dict (as produced by
                :class:`~finsage.data.chunker.FilingChunker`), optionally
                carrying a ``metadata`` mapping.
            task_type: One of :data:`TASK_TYPES`.
            question: For ``filing_qa``, the question to append to the
                instruction. Ignored for other task types.
            output: The target answer. Left empty in Phase 1 and filled in
                Phase 4; never fabricated.

        Returns:
            A dict with ``id``, ``source``, ``instruction``, ``input``,
            ``output``, ``task_type``, and ``metadata`` keys.

        Raises:
            ValueError: If ``task_type`` is not a supported task type.
        """
        if task_type not in self.task_types:
            raise ValueError(f"Unknown task_type {task_type!r}; expected one of {self.task_types}")

        metadata: dict[str, Any] = dict(chunk.get("metadata", {}))
        instruction = _INSTRUCTION_TEMPLATES[task_type]
        if task_type == "filing_qa" and question:
            instruction = f"{instruction}\n\nQuestion: {question}"

        ticker = metadata.get("ticker", "UNK")
        year = metadata.get("year", "XXXX")
        filing_type = metadata.get("filing_type", "NA")
        section = metadata.get("section", "")
        source = f"{ticker} {year} {filing_type} {section}".strip()

        example = {
            "id": _make_example_id(metadata, task_type, chunk.get("chunk_id", 0)),
            "source": source,
            "instruction": instruction,
            "input": str(chunk.get("text", "")),
            "output": output,
            "task_type": task_type,
            "metadata": metadata,
        }
        return example
