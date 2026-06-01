"""Selection and Markdown rendering of qualitative before/after examples.

Picks a small, representative set of side-by-side examples (best improvements, a
regression, an average case, and a faithfulness case) and renders them so a
non-engineer can read them. All text is safely truncated so large filing
excerpts never bloat the report.
"""

from __future__ import annotations

from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: Maximum characters rendered for any single free-text field.
_MAX_FIELD_CHARS = 600
#: Maximum characters rendered for the (already short) filing excerpt preview.
_MAX_EXCERPT_CHARS = 300


def _truncate(text: Any, limit: int = _MAX_FIELD_CHARS) -> str:
    """Safely stringify and truncate ``text`` to ``limit`` characters."""
    if text is None:
        return "N/A"
    s = str(text).strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _mean_improvement(row: dict[str, Any]) -> float:
    """Mean of the per-metric improvement deltas for a qualitative row."""
    summary = row.get("improvement_summary") or {}
    deltas = [float(v) for v in summary.values() if isinstance(v, (int, float))]
    return sum(deltas) / len(deltas) if deltas else 0.0


def _min_improvement(row: dict[str, Any]) -> float:
    """Smallest per-metric improvement delta (most negative regression) for a row."""
    summary = row.get("improvement_summary") or {}
    deltas = [float(v) for v in summary.values() if isinstance(v, (int, float))]
    return min(deltas) if deltas else 0.0


def _is_faithfulness_case(row: dict[str, Any]) -> bool:
    """True if a row relates to faithfulness/hallucination."""
    if row.get("task_type") == "hallucination_detection":
        return True
    summary = row.get("improvement_summary") or {}
    return any("faithful" in k or "hallucin" in k for k in summary)


def select_qualitative_examples(
    qualitative_rows: list[dict[str, Any]],
    max_examples: int = 5,
) -> list[dict[str, Any]]:
    """Choose a representative subset of qualitative examples.

    Selection favours, in order: the two largest improvements, one regression /
    failure case, one average case, and one faithfulness-related case. Each
    returned row is annotated with a ``"category"`` key. Duplicates are avoided
    and the result is capped at ``max_examples``.

    Args:
        qualitative_rows: Rows from ``qualitative_comparisons.jsonl``.
        max_examples: Maximum number of examples to return.

    Returns:
        A list of selected rows (each with a ``"category"`` label), possibly
        shorter than ``max_examples`` when input is sparse.
    """
    if not qualitative_rows:
        return []

    scored = sorted(qualitative_rows, key=_mean_improvement, reverse=True)
    chosen: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    def take(row: dict[str, Any] | None, category: str) -> None:
        if row is None or id(row) in used_ids or len(chosen) >= max_examples:
            return
        annotated = dict(row)
        annotated["category"] = category
        chosen.append(annotated)
        used_ids.add(id(row))

    # Two best improvements.
    for row in scored[:2]:
        if _mean_improvement(row) > 0:
            take(row, "improvement")

    # One regression / failure case: the row whose worst single metric dropped
    # the most (any metric regressing counts, even if the mean improved).
    worst = min(scored, key=_min_improvement) if scored else None
    if worst is not None and _min_improvement(worst) < 0:
        take(worst, "regression")

    # One average case (median by mean improvement).
    if scored:
        take(scored[len(scored) // 2], "average")

    # One faithfulness / hallucination case.
    faithful = next((r for r in scored if _is_faithfulness_case(r) and id(r) not in used_ids), None)
    take(faithful, "faithfulness")

    # Backfill with remaining highest-scoring examples if room remains.
    for row in scored:
        if len(chosen) >= max_examples:
            break
        take(row, "additional")

    return chosen[:max_examples]


#: Human-readable labels for selection categories.
_CATEGORY_LABELS = {
    "improvement": "Improvement",
    "regression": "Regression / failure case",
    "average": "Average case",
    "faithfulness": "Faithfulness / hallucination case",
    "additional": "Additional example",
}


def _takeaway(example: dict[str, Any]) -> str:
    """Produce a one-line takeaway for an example based on its category/deltas."""
    category = example.get("category", "additional")
    mean = _mean_improvement(example)
    if category == "regression" or mean < 0:
        return (
            "Fine-tuning regressed on lexical-overlap metrics here, often because the "
            "fine-tuned answer adds correct detail not present in the short reference."
        )
    if category == "faithfulness":
        return "Fine-tuning improved grounding/faithfulness relative to the base model."
    if mean > 0:
        return "Fine-tuned answer is more specific and better grounded in the excerpt."
    return "Base and fine-tuned answers are comparable on this example."


def format_qualitative_example(example: dict[str, Any], index: int) -> str:
    """Render a single qualitative example as a Markdown block.

    Args:
        example: A selected qualitative row (ideally with a ``"category"``).
        index: 1-based index used in the heading.

    Returns:
        A Markdown string for the example.
    """
    label = _CATEGORY_LABELS.get(example.get("category", "additional"), "Example")
    task = str(example.get("task_type", "unknown")).replace("_", " ")
    summary = example.get("improvement_summary") or {}
    delta_str = (
        ", ".join(
            f"{k.replace('_', ' ')}: {v:+.3f}"
            for k, v in summary.items()
            if isinstance(v, (int, float))
        )
        or "N/A"
    )

    lines = [
        f"#### Example {index} — {label} ({task})",
        "",
        f"**Instruction:** {_truncate(example.get('instruction'), 200)}",
        "",
        f"**Filing excerpt:** {_truncate(example.get('input_preview'), _MAX_EXCERPT_CHARS)}",
        "",
        f"**Reference answer:** {_truncate(example.get('reference'))}",
        "",
        f"**Base model:** {_truncate(example.get('baseline_prediction'))}",
        "",
        f"**Fine-tuned (FinSage-7B):** {_truncate(example.get('finetuned_prediction'))}",
        "",
        f"**Metric change:** {delta_str}",
        "",
        f"**Takeaway:** {_takeaway(example)}",
    ]
    return "\n".join(lines)


def build_qualitative_section(examples: list[dict[str, Any]]) -> str:
    """Render the full qualitative-examples section.

    Args:
        examples: Selected qualitative rows.

    Returns:
        A Markdown string, or a "not available" note when there are none.
    """
    if not examples:
        return "_No qualitative examples available._"
    blocks = [format_qualitative_example(ex, i) for i, ex in enumerate(examples, start=1)]
    return "\n\n".join(blocks)
