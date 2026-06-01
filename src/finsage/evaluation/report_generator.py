"""Render evaluation results into a Markdown benchmark report.

This module is dependency-free and fully implemented in Phase 1: given two
result mappings it produces the base-vs-FinSage comparison table used in
``reports/benchmark_report.md``. The numbers are supplied by the eval runner in
later phases — nothing here fabricates metrics.
"""

from __future__ import annotations

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


def build_comparison_table(
    base_results: dict[str, float],
    finetuned_results: dict[str, float],
    metric_labels: dict[str, str] | None = None,
) -> str:
    """Build a Markdown table comparing base and fine-tuned metrics.

    Args:
        base_results: Mapping of metric key to value for the base model.
        finetuned_results: Mapping of metric key to value for FinSage-7B.
        metric_labels: Optional human-readable label per metric key.

    Returns:
        A Markdown table string with a Delta column. Metrics present in either
        mapping are included; missing values render as ``n/a``.
    """
    labels = metric_labels or {}
    keys = sorted(set(base_results) | set(finetuned_results))

    lines = [
        "| Metric | Base Mistral-7B | FinSage-7B | Delta |",
        "|--------|-----------------|------------|-------|",
    ]
    for key in keys:
        label = labels.get(key, key)
        base = base_results.get(key)
        fine = finetuned_results.get(key)
        delta = f"{fine - base:+.3f}" if base is not None and fine is not None else "n/a"
        base_str = f"{base:.3f}" if base is not None else "n/a"
        fine_str = f"{fine:.3f}" if fine is not None else "n/a"
        lines.append(f"| {label} | {base_str} | {fine_str} | {delta} |")

    logger.debug("Built comparison table for %d metrics", len(keys))
    return "\n".join(lines)


def render_report(
    base_results: dict[str, float],
    finetuned_results: dict[str, float],
    metric_labels: dict[str, str] | None = None,
    title: str = "FinSage-7B Benchmark Report",
) -> str:
    """Render a full Markdown benchmark report.

    Args:
        base_results: Base-model metric values.
        finetuned_results: FinSage-7B metric values.
        metric_labels: Optional human-readable labels per metric key.
        title: Report title.

    Returns:
        The full Markdown document as a string.
    """
    table = build_comparison_table(base_results, finetuned_results, metric_labels)
    return (
        f"# {title}\n\n"
        "Base Mistral-7B-Instruct vs the QLoRA fine-tuned FinSage-7B on the "
        "held-out test split.\n\n"
        f"{table}\n\n"
        "> Metrics are produced by the evaluation harness; verify against the "
        "raw result files before publishing.\n"
    )
