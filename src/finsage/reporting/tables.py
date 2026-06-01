"""Markdown table builders for the benchmark report.

All formatters are defensive: ``None``, non-numeric, ``NaN`` and ``Infinity``
values render as ``"N/A"`` (never as ``nan``/``inf``), and positive deltas are
prefixed with ``+`` for at-a-glance readability in both Markdown and PDF.
"""

from __future__ import annotations

import math
from typing import Any, TypeGuard

#: Placeholder rendered whenever a value is missing or not finite.
NA = "N/A"


def _is_finite_number(value: Any) -> TypeGuard[float]:
    """Return ``True`` if ``value`` is a finite int/float (not bool/NaN/Inf)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a GitHub-flavoured Markdown table.

    Args:
        headers: Column headers.
        rows: Row values; each cell is stringified (``None`` becomes ``"N/A"``).

    Returns:
        The table as a Markdown string. If there are no rows, a single
        ``"_No data available._"`` line is returned instead.
    """
    if not headers:
        return "_No data available._"
    if not rows:
        return "_No data available._"

    def cell(value: Any) -> str:
        if value is None:
            return NA
        return str(value).replace("|", "\\|").replace("\n", " ")

    header_line = "| " + " | ".join(cell(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def format_metric(value: Any, decimals: int = 3) -> str:
    """Format a metric value to fixed decimals, or ``"N/A"`` if not finite.

    Args:
        value: The metric value.
        decimals: Number of decimal places.

    Returns:
        A formatted string, never ``nan``/``inf``.
    """
    if not _is_finite_number(value):
        return NA
    return f"{float(value):.{decimals}f}"


def format_delta(delta: Any, decimals: int = 3) -> str:
    """Format a signed delta with an explicit ``+`` for non-negative values.

    Args:
        delta: The delta value.
        decimals: Number of decimal places.

    Returns:
        A signed string (e.g. ``"+0.200"`` / ``"-0.100"``), or ``"N/A"``.
    """
    if not _is_finite_number(delta):
        return NA
    val = float(delta)
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{decimals}f}"


def _improvement_marker(entry: dict[str, Any]) -> str:
    """Return a small up/down/flat marker for a comparison entry."""
    delta = entry.get("absolute_delta")
    if not _is_finite_number(delta):
        return NA
    val = float(delta)
    if val > 0:
        return "▲ improved"
    if val < 0:
        return "▼ regressed"
    return "— no change"


def build_overall_metrics_table(comparison: dict[str, Any] | None) -> str:
    """Build the overall base-vs-fine-tuned metrics table.

    Args:
        comparison: A ``comparison_results`` dict with an ``overall_comparison``
            mapping of metric name to ``{baseline, finetuned, absolute_delta,
            relative_delta_pct, improved}``.

    Returns:
        A Markdown table, or a "not available" note.
    """
    if not comparison:
        return "_Overall comparison not available._"
    overall = comparison.get("overall_comparison") or {}
    if not overall:
        return "_Overall comparison not available._"

    rows: list[list[Any]] = []
    for metric in sorted(overall):
        entry = overall[metric] or {}
        rows.append(
            [
                metric.replace("_", " "),
                format_metric(entry.get("baseline")),
                format_metric(entry.get("finetuned")),
                format_delta(entry.get("absolute_delta")),
                format_delta(entry.get("relative_delta_pct"), decimals=1),
                _improvement_marker(entry),
            ]
        )
    return markdown_table(
        ["Metric", "Base", "Fine-tuned", "Δ abs", "Δ %", "Result"],
        rows,
    )


def build_task_metrics_table(comparison: dict[str, Any] | None) -> str:
    """Build the per-task metric-delta table.

    Args:
        comparison: Either a ``comparison_results`` dict (with
            ``by_task_comparison``) or a ``metric_delta_by_task`` mapping of
            ``task -> metric -> {...}``.

    Returns:
        A Markdown table, or a "not available" note.
    """
    if not comparison:
        return "_Per-task comparison not available._"
    by_task = comparison.get("by_task_comparison") if "by_task_comparison" in comparison else None
    if by_task is None:
        # Treat the whole dict as a metric_delta_by_task mapping.
        by_task = comparison
    if not by_task:
        return "_Per-task comparison not available._"

    rows: list[list[Any]] = []
    for task in sorted(by_task):
        metrics = by_task[task] or {}
        for metric in sorted(metrics):
            entry = metrics[metric]
            if not isinstance(entry, dict):
                continue
            rows.append(
                [
                    task.replace("_", " "),
                    metric.replace("_", " "),
                    format_metric(entry.get("baseline")),
                    format_metric(entry.get("finetuned")),
                    format_delta(entry.get("absolute_delta")),
                    _improvement_marker(entry),
                ]
            )
    return markdown_table(
        ["Task", "Metric", "Base", "Fine-tuned", "Δ abs", "Result"],
        rows,
    )


def build_dataset_stats_table(dataset_stats: dict[str, Any] | None) -> str:
    """Build a dataset-statistics table (totals, splits, task distribution).

    Args:
        dataset_stats: A ``dataset_stats.json`` dict.

    Returns:
        A Markdown table, or a "not available" note.
    """
    if not dataset_stats:
        return "_Dataset statistics not available._"

    rows: list[list[Any]] = [
        ["Total examples", dataset_stats.get("total_examples", NA)],
        ["Avg. input length (chars)", format_metric(dataset_stats.get("average_input_length"), 1)],
        [
            "Avg. output length (chars)",
            format_metric(dataset_stats.get("average_output_length"), 1),
        ],
    ]
    for split, count in (dataset_stats.get("examples_per_split") or {}).items():
        rows.append([f"Split · {split}", count])
    for task, count in sorted((dataset_stats.get("examples_per_task_type") or {}).items()):
        rows.append([f"Task · {task.replace('_', ' ')}", count])

    return markdown_table(["Statistic", "Value"], rows)


def build_training_summary_table(training_summary: dict[str, Any] | None) -> str:
    """Build a QLoRA training-configuration table.

    Args:
        training_summary: A ``training_summary.json`` dict.

    Returns:
        A Markdown table, or a "not available" note.
    """
    if not training_summary:
        return "_Training summary not available (model not yet fine-tuned)._"

    field_labels = [
        ("model_id", "Base model"),
        ("num_train_examples", "Train examples"),
        ("num_eval_examples", "Eval examples"),
        ("lora_r", "LoRA rank (r)"),
        ("lora_alpha", "LoRA alpha"),
        ("learning_rate", "Learning rate"),
        ("epochs", "Epochs"),
        ("max_seq_length", "Max sequence length"),
        ("packing", "Sequence packing"),
        ("final_train_loss", "Final train loss"),
        ("final_eval_loss", "Final eval loss"),
    ]
    rows: list[list[Any]] = []
    for key, label in field_labels:
        value = training_summary.get(key)
        rows.append([label, NA if value is None else value])
    return markdown_table(["Setting", "Value"], rows)


def build_latency_table(latency_results: dict[str, Any] | None) -> str:
    """Build a serving-latency table from a latency benchmark dict.

    Args:
        latency_results: A latency benchmark dict with ``p50/p95/p99_latency_s``.

    Returns:
        A Markdown table, or a "not available" note.
    """
    if not latency_results:
        return "_Latency benchmark not available._"

    rows: list[list[Any]] = [
        ["Endpoint", latency_results.get("endpoint", NA)],
        ["Model", latency_results.get("model", NA)],
        ["Requests", latency_results.get("num_requests", NA)],
        ["Concurrency", latency_results.get("concurrency", NA)],
        ["Successful", latency_results.get("successful_requests", NA)],
        ["Failed", latency_results.get("failed_requests", NA)],
        ["p50 latency (s)", format_metric(latency_results.get("p50_latency_s"))],
        ["p95 latency (s)", format_metric(latency_results.get("p95_latency_s"))],
        ["p99 latency (s)", format_metric(latency_results.get("p99_latency_s"))],
        ["Avg latency (s)", format_metric(latency_results.get("avg_latency_s"))],
        ["Approx tokens/s", format_metric(latency_results.get("approx_tokens_per_second"), 1)],
    ]
    return markdown_table(["Metric", "Value"], rows)


#: Known limitations of the FinSage-7B benchmark, surfaced honestly in-report.
_LIMITATIONS: list[tuple[str, str]] = [
    (
        "Weak-supervision targets",
        "Phase 3 instruction targets are template/extractive (no LLM teacher), so "
        "reference answers approximate, not certify, ground truth.",
    ),
    (
        "Lexical faithfulness proxy",
        "The default faithfulness metric is lexical-overlap based; optional NLI "
        "entailment is available but off by default. Neither is a full hallucination audit.",
    ),
    (
        "Small evaluation set",
        "The held-out evaluation set is small; metric deltas have wide confidence "
        "intervals and should be read as directional, not definitive.",
    ),
    (
        "Single base model",
        "Only Mistral-7B-Instruct is compared; results may not transfer to other "
        "base models or larger scales.",
    ),
    (
        "No live market data",
        "The model reasons only over the provided filing excerpt; it has no access "
        "to prices, real-time disclosures, or post-filing events.",
    ),
    (
        "Not investment advice",
        "Outputs are informational only and must not be used for investment decisions.",
    ),
]


def build_limitations_table() -> str:
    """Build the static limitations table.

    Returns:
        A Markdown table summarising known limitations.
    """
    rows = [[name, detail] for name, detail in _LIMITATIONS]
    return markdown_table(["Limitation", "Detail"], rows)
