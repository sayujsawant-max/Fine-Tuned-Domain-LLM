"""Build instruction-tuning examples from filing chunks.

Phase 3 generates **deterministic, template/extractive** outputs — no LLM APIs
are used. These are *weak-supervision* targets, not human-written gold answers;
every example is flagged as such in its metadata (``generation_method`` and
``weak_supervision``). The goal is a reproducible first supervised dataset that
later phases can improve with human review or LLM-assisted generation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: The ten supported instruction task types, in blueprint order.
TASK_TYPES: list[str] = [
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
]

# --- keyword banks -----------------------------------------------------------
_YOY_KEYWORDS = (
    "increase",
    "increased",
    "decrease",
    "decreased",
    "grew",
    "growth",
    "declined",
    "compared with",
    "compared to",
    "year over year",
    "year-over-year",
    "prior year",
    "fiscal year",
    "percentage",
    "basis points",
)
_RISK_KEYWORDS = (
    "risk",
    "risks",
    "uncertainty",
    "competition",
    "regulation",
    "supply chain",
    "cybersecurity",
    "litigation",
    "market",
    "inflation",
    "demand",
    "dependency",
    "disruption",
)
_REVENUE_KEYWORDS = (
    "revenue",
    "sales",
    "demand",
    "growth",
    "customer",
    "product",
    "segment",
    "pricing",
    "volume",
    "subscription",
    "services",
    "margin",
)
_POSITIVE_TERMS = (
    "growth",
    "increased",
    "improved",
    "strong",
    "higher",
    "expansion",
    "profitability",
    "demand",
)
_NEGATIVE_TERMS = (
    "risk",
    "decline",
    "decreased",
    "lower",
    "uncertainty",
    "inflation",
    "litigation",
    "weakness",
    "disruption",
    "loss",
)

_UNSUPPORTED_CLAIM = "The company guarantees future investment returns."

# --- regex -------------------------------------------------------------------
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s?%")
_MAGNITUDE_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion|thousand)", re.IGNORECASE
)
_BPS_RE = re.compile(r"\d+(?:\.\d+)?\s?basis points", re.IGNORECASE)


def _sentences(text: str, min_words: int = 3) -> list[str]:
    """Split text into meaningful sentences.

    Args:
        text: Source text.
        min_words: Minimum words for a sentence to be considered meaningful.

    Returns:
        A list of trimmed sentences with at least ``min_words`` words.
    """
    parts = (s.strip() for s in _SENTENCE_RE.split(text or ""))
    return [s for s in parts if len(s.split()) >= min_words]


def _extractive_summary(text: str, max_sentences: int = 4) -> str:
    """Return the first few meaningful sentences as an extractive summary.

    Args:
        text: Source text.
        max_sentences: Maximum number of sentences to include.

    Returns:
        A non-empty extractive summary (falls back to a text prefix).
    """
    sents = _sentences(text)
    if sents:
        return " ".join(sents[:max_sentences])
    snippet = (text or "").strip()[:300]
    return snippet or "(no extractable content in excerpt)"


def _sentences_with_keywords(text: str, keywords: tuple[str, ...], limit: int = 5) -> list[str]:
    """Return sentences containing any of the given keywords.

    Args:
        text: Source text.
        keywords: Lower-case keywords/phrases to match.
        limit: Maximum number of sentences to return.

    Returns:
        Matching sentences, up to ``limit``.
    """
    out: list[str] = []
    for sentence in _sentences(text):
        low = sentence.lower()
        if any(kw in low for kw in keywords):
            out.append(sentence)
        if len(out) >= limit:
            break
    return out


def _find_metrics(text: str) -> list[str]:
    """Extract monetary values, percentages, magnitudes, and basis points.

    Args:
        text: Source text.

    Returns:
        A de-duplicated list of metric strings, in first-seen order.
    """
    found: list[str] = []
    for pattern in (_MONEY_RE, _PERCENT_RE, _MAGNITUDE_RE, _BPS_RE):
        for match in pattern.findall(text or ""):
            value = match.strip()
            if value and value not in found:
                found.append(value)
    return found


def _bullets(items: list[str]) -> str:
    """Render items as a Markdown bullet list.

    Args:
        items: Bullet contents.

    Returns:
        A newline-joined bullet list.
    """
    return "\n".join(f"- {item}" for item in items)


class InstructionBuilder:
    """Builds template/extractive instruction examples from filing chunks."""

    task_types: list[str] = TASK_TYPES

    def validate_task_type(self, task_type: str) -> None:
        """Validate that ``task_type`` is supported.

        Args:
            task_type: The task type to check.

        Raises:
            ValueError: If ``task_type`` is not in :data:`TASK_TYPES`.
        """
        if task_type not in self.task_types:
            raise ValueError(f"Unknown task_type {task_type!r}; expected one of {self.task_types}")

    @staticmethod
    def _example_id(metadata: dict[str, Any], chunk_id: Any, task_type: str) -> str:
        """Build a stable, unique example id.

        Args:
            metadata: Chunk metadata (ticker/cik/year/form/section/accession).
            chunk_id: The originating chunk's identifier.
            task_type: The task type for this example.

        Returns:
            A hyphen-joined id, e.g.
            ``AAPL-2022-10-K-000..108-risk_factors-0-risk_summary``.
        """
        ticker = str(metadata.get("ticker") or metadata.get("cik") or "UNK").upper()
        year = str(metadata.get("year") or str(metadata.get("filing_date", ""))[:4] or "XXXX")
        form = str(metadata.get("form", "NA")).replace("/", "-")
        accession = str(metadata.get("accession_number_no_dashes", "") or "")[-6:]
        section = str(metadata.get("section", "sec"))
        return "-".join(
            part
            for part in [ticker, year, form, accession, section, str(chunk_id), task_type]
            if part
        )

    def build_example(
        self,
        chunk: dict[str, Any],
        task_type: str,
        example_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a single JSONL-ready instruction example.

        Args:
            chunk: A chunk dict from
                :class:`~finsage.data.chunker.FilingChunker`.
            task_type: One of :data:`TASK_TYPES`.
            example_id: Explicit id; auto-generated when ``None``.

        Returns:
            A dict with ``id``, ``instruction``, ``input``, ``output``,
            ``task_type``, ``source``, and ``metadata`` keys.

        Raises:
            ValueError: If ``task_type`` is not supported.
        """
        self.validate_task_type(task_type)
        meta = dict(chunk.get("metadata", {}))
        chunk_id = chunk.get("chunk_id", 0)
        text = str(chunk.get("text", ""))

        instruction, input_text, output = self._render(task_type, text, chunk_id)

        ticker = meta.get("ticker") or meta.get("cik") or "UNK"
        year = meta.get("year") or str(meta.get("filing_date", ""))[:4] or "XXXX"
        form = meta.get("form", "NA")
        section = meta.get("section", "")
        source = " ".join(str(x) for x in (ticker, year, form, section) if x)

        example_meta = {
            **meta,
            "chunk_id": chunk_id,
            "token_count": chunk.get("token_count"),
            "generation_method": "template_extractive",
            "weak_supervision": True,
        }
        return {
            "id": example_id or self._example_id(meta, chunk_id, task_type),
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "task_type": task_type,
            "source": source,
            "metadata": example_meta,
        }

    def build_examples_for_chunk(
        self,
        chunk: dict[str, Any],
        task_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build one example per requested task type for a chunk.

        Args:
            chunk: A chunk dict.
            task_types: Task types to build; defaults to all :data:`TASK_TYPES`.

        Returns:
            A list of instruction examples.
        """
        selected = task_types or self.task_types
        return [self.build_example(chunk, task_type) for task_type in selected]

    # ------------------------------------------------------------------
    # Per-task rendering
    # ------------------------------------------------------------------
    def _render(self, task_type: str, text: str, chunk_id: Any) -> tuple[str, str, str]:
        """Render the (instruction, input, output) triple for a task type.

        Args:
            task_type: The task type to render.
            text: The chunk text.
            chunk_id: The chunk id (used to vary hallucination polarity).

        Returns:
            A ``(instruction, input, output)`` tuple.
        """
        renderer = getattr(self, f"_render_{task_type}")
        return renderer(text, chunk_id)  # type: ignore[no-any-return]

    def _render_risk_summary(self, text: str, _chunk_id: Any) -> tuple[str, str, str]:
        instruction = "Summarize the key risk factors discussed in the filing excerpt."
        return instruction, text, _extractive_summary(text)

    def _render_mda_explanation(self, text: str, _chunk_id: Any) -> tuple[str, str, str]:
        instruction = "Explain the main management discussion points in this MD&A excerpt."
        return instruction, text, _extractive_summary(text)

    def _render_metric_extraction(self, text: str, _chunk_id: Any) -> tuple[str, str, str]:
        instruction = (
            "Extract financial metrics, figures, percentages, or monetary values "
            "mentioned in the filing excerpt."
        )
        metrics = _find_metrics(text)
        output = (
            _bullets(metrics)
            if metrics
            else "No explicit financial metric was found in the provided excerpt."
        )
        return instruction, text, output

    def _render_yoy_comparison(self, text: str, _chunk_id: Any) -> tuple[str, str, str]:
        instruction = (
            "Identify any year-over-year comparison or period-over-period change "
            "mentioned in the filing excerpt."
        )
        sentences = _sentences_with_keywords(text, _YOY_KEYWORDS)
        output = (
            _bullets(sentences)
            if sentences
            else (
                "No explicit year-over-year or period-over-period comparison was "
                "found in the provided excerpt."
            )
        )
        return instruction, text, output

    def _render_business_risk_identification(
        self, text: str, _chunk_id: Any
    ) -> tuple[str, str, str]:
        instruction = (
            "Identify business risks or operational risks mentioned in the filing excerpt."
        )
        sentences = _sentences_with_keywords(text, _RISK_KEYWORDS)
        output = (
            _bullets(sentences)
            if sentences
            else "No explicit business or operational risk was found in the provided excerpt."
        )
        return instruction, text, output

    def _render_revenue_driver_explanation(self, text: str, _chunk_id: Any) -> tuple[str, str, str]:
        instruction = (
            "Identify revenue drivers or business performance drivers mentioned "
            "in the filing excerpt."
        )
        sentences = _sentences_with_keywords(text, _REVENUE_KEYWORDS)
        output = (
            _bullets(sentences)
            if sentences
            else "No explicit revenue driver was found in the provided excerpt."
        )
        return instruction, text, output

    def _render_filing_qa(self, text: str, _chunk_id: Any) -> tuple[str, str, str]:
        instruction = (
            "Answer the following filing-grounded question using only the provided "
            "excerpt: What is the main point of this excerpt?"
        )
        return instruction, text, _extractive_summary(text, max_sentences=3)

    def _render_analyst_summary(self, text: str, _chunk_id: Any) -> tuple[str, str, str]:
        instruction = "Write a concise analyst-style summary of the filing excerpt."
        sents = _sentences(text)
        summary = " ".join(sents[:3]) if sents else _extractive_summary(text)
        key_point = sents[0] if sents else summary
        evidence = (text or "").strip()[:200]
        output = f"Analyst Summary: {summary}\n" f"Key Point: {key_point}\n" f"Evidence: {evidence}"
        return instruction, text, output

    def _render_outlook_classification(self, text: str, _chunk_id: Any) -> tuple[str, str, str]:
        instruction = (
            "Classify the financial or business outlook in the excerpt as positive, "
            "neutral, or negative. Explain briefly."
        )
        low = (text or "").lower()
        pos = sum(low.count(term) for term in _POSITIVE_TERMS)
        neg = sum(low.count(term) for term in _NEGATIVE_TERMS)
        if pos > neg:
            label = "positive"
        elif neg > pos:
            label = "negative"
        else:
            label = "neutral"
        reason = (
            f"Found {pos} positive and {neg} negative outlook term(s) in the excerpt "
            "(template keyword scoring)."
        )
        output = json.dumps({"label": label, "reason": reason})
        return instruction, text, output

    def _render_hallucination_detection(self, text: str, chunk_id: Any) -> tuple[str, str, str]:
        instruction = "Determine whether the proposed answer is supported by the filing excerpt."
        sents = _sentences(text)
        supported = (int(chunk_id) % 2 == 0) if str(chunk_id).lstrip("-").isdigit() else True
        if supported and sents:
            claim = sents[0]
            reason = "The proposed answer is a sentence taken directly from the excerpt."
            is_supported = True
        else:
            claim = _UNSUPPORTED_CLAIM
            reason = "The proposed answer is a generic claim not stated in the excerpt."
            is_supported = False
        input_text = f"{text}\n\nProposed answer: {claim}"
        output = json.dumps({"supported": is_supported, "reason": reason})
        return instruction, input_text, output
