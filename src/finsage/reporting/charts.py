"""Chart generation for the benchmark report (matplotlib only, lazy import).

Every function is optional and robust: if matplotlib is not installed, or the
required data is missing, the chart is skipped (a warning is logged) and the
overall report still builds. Charts are saved as PNG into the figures directory.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


def _import_pyplot() -> Any | None:
    """Import ``matplotlib.pyplot`` with a headless backend, or return ``None``.

    Returns:
        The ``pyplot`` module, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display required
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - import environment dependent
        logger.warning("matplotlib unavailable; skipping charts: %s", exc)
        return None


def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _save(plt: Any, fig: Any, output_path: Path | str) -> Path:
    """Tighten layout, save ``fig`` to ``output_path``, and close it."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote chart %s", out)
    return out


def create_overall_metrics_chart(
    comparison: dict[str, Any] | None, output_path: Path | str
) -> Path | None:
    """Grouped bar chart comparing base vs fine-tuned on overall metrics.

    Args:
        comparison: A ``comparison_results`` dict with ``overall_comparison``.
        output_path: Destination PNG path.

    Returns:
        The written path, or ``None`` if skipped.
    """
    plt = _import_pyplot()
    if plt is None:
        return None
    overall = (comparison or {}).get("overall_comparison") or {}
    metrics, base_vals, ft_vals = [], [], []
    for metric in sorted(overall):
        entry = overall[metric] or {}
        b, f = _finite(entry.get("baseline")), _finite(entry.get("finetuned"))
        if b is None and f is None:
            continue
        metrics.append(metric.replace("_", "\n"))
        base_vals.append(b or 0.0)
        ft_vals.append(f or 0.0)
    if not metrics:
        logger.warning("No overall metrics to chart; skipping overall_metrics chart.")
        return None

    x = range(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(metrics) * 1.1), 4.5))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Base", color="#9aa4b2")
    ax.bar([i + width / 2 for i in x], ft_vals, width, label="Fine-tuned", color="#3b82f6")
    ax.set_title("Overall metrics: base vs fine-tuned")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    return _save(plt, fig, output_path)


def create_task_delta_chart(
    comparison: dict[str, Any] | None, output_path: Path | str
) -> Path | None:
    """Horizontal bar chart of mean absolute metric delta per task.

    Args:
        comparison: A ``comparison_results`` dict (``by_task_comparison``) or a
            ``metric_delta_by_task`` mapping.
        output_path: Destination PNG path.

    Returns:
        The written path, or ``None`` if skipped.
    """
    plt = _import_pyplot()
    if plt is None:
        return None
    by_task = (comparison or {}).get("by_task_comparison")
    if by_task is None:
        by_task = comparison or {}

    tasks, deltas = [], []
    for task in sorted(by_task):
        metrics = by_task[task] or {}
        vals: list[float] = []
        for m in metrics.values():
            if isinstance(m, dict):
                d = _finite(m.get("absolute_delta"))
                if d is not None:
                    vals.append(d)
        if not vals:
            continue
        tasks.append(task.replace("_", " "))
        deltas.append(sum(vals) / len(vals))
    if not tasks:
        logger.warning("No per-task deltas to chart; skipping task_delta chart.")
        return None

    colors = ["#3b82f6" if d >= 0 else "#ef4444" for d in deltas]
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(tasks) * 0.45)))
    ax.barh(tasks, deltas, color=colors)
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_title("Mean absolute metric delta by task (fine-tuned − base)")
    ax.set_xlabel("Δ (positive = improvement)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    return _save(plt, fig, output_path)


def create_dataset_distribution_chart(
    dataset_stats: dict[str, Any] | None, output_path: Path | str
) -> Path | None:
    """Bar chart of instruction examples per task type.

    Args:
        dataset_stats: A ``dataset_stats.json`` dict.
        output_path: Destination PNG path.

    Returns:
        The written path, or ``None`` if skipped.
    """
    plt = _import_pyplot()
    if plt is None:
        return None
    per_task = (dataset_stats or {}).get("examples_per_task_type") or {}
    items = [(k, v) for k, v in per_task.items() if _finite(v) is not None]
    if not items:
        logger.warning("No dataset task distribution to chart; skipping.")
        return None
    items.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k.replace("_", " ") for k, _ in items]
    counts = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(labels) * 0.4)))
    ax.barh(labels[::-1], counts[::-1], color="#10b981")
    ax.set_title("Instruction examples per task type")
    ax.set_xlabel("Examples")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    return _save(plt, fig, output_path)


def create_latency_chart(
    latency_results: dict[str, Any] | None, output_path: Path | str
) -> Path | None:
    """Bar chart of p50/p95/p99 serving latency.

    Args:
        latency_results: A latency benchmark dict.
        output_path: Destination PNG path.

    Returns:
        The written path, or ``None`` if skipped.
    """
    plt = _import_pyplot()
    if plt is None:
        return None
    labels, values = [], []
    for pct in ("p50", "p95", "p99"):
        v = _finite((latency_results or {}).get(f"{pct}_latency_s"))
        if v is not None:
            labels.append(pct)
            values.append(v)
    if not values:
        logger.warning("No latency percentiles to chart; skipping latency chart.")
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color="#8b5cf6")
    ax.set_title("Serving latency percentiles")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    return _save(plt, fig, output_path)


#: Metric names treated as faithfulness / hallucination-related.
_FAITHFULNESS_METRICS = (
    "lexical_faithfulness",
    "nli_faithfulness",
    "classification_accuracy",
)


def create_hallucination_chart(
    comparison: dict[str, Any] | None, output_path: Path | str
) -> Path | None:
    """Bar chart of faithfulness / hallucination-related metrics (base vs fine-tuned).

    Args:
        comparison: A ``comparison_results`` dict with ``overall_comparison``.
        output_path: Destination PNG path.

    Returns:
        The written path, or ``None`` if skipped.
    """
    plt = _import_pyplot()
    if plt is None:
        return None
    overall = (comparison or {}).get("overall_comparison") or {}
    metrics, base_vals, ft_vals = [], [], []
    for metric in _FAITHFULNESS_METRICS:
        entry = overall.get(metric)
        if not isinstance(entry, dict):
            continue
        b, f = _finite(entry.get("baseline")), _finite(entry.get("finetuned"))
        if b is None and f is None:
            continue
        metrics.append(metric.replace("_", "\n"))
        base_vals.append(b or 0.0)
        ft_vals.append(f or 0.0)
    if not metrics:
        logger.warning("No faithfulness metrics to chart; skipping hallucination chart.")
        return None

    x = range(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(5, len(metrics) * 1.6), 4.5))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Base", color="#9aa4b2")
    ax.bar([i + width / 2 for i in x], ft_vals, width, label="Fine-tuned", color="#3b82f6")
    ax.set_title("Faithfulness / hallucination-related metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    return _save(plt, fig, output_path)


def create_all_report_charts(inputs: dict[str, Any], output_dir: Path | str) -> dict[str, str]:
    """Generate every report chart that has data, skipping the rest.

    Args:
        inputs: The loaded report inputs (see ``load_optional_report_inputs``).
        output_dir: Directory to write PNG charts into.

    Returns:
        A mapping of chart name to the written PNG path (only successful charts).
    """
    out_dir = Path(output_dir)
    comparison = inputs.get("comparison_results")
    delta_by_task = inputs.get("metric_delta_by_task") or comparison
    dataset_stats = inputs.get("dataset_stats")
    latency = inputs.get("api_latency") or inputs.get("vllm_latency")

    jobs = {
        "overall_metrics": lambda p: create_overall_metrics_chart(comparison, p),
        "task_delta": lambda p: create_task_delta_chart(delta_by_task, p),
        "dataset_distribution": lambda p: create_dataset_distribution_chart(dataset_stats, p),
        "latency": lambda p: create_latency_chart(latency, p),
        "hallucination": lambda p: create_hallucination_chart(comparison, p),
    }

    created: dict[str, str] = {}
    for name, fn in jobs.items():
        try:
            path = fn(out_dir / f"report_{name}.png")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Chart %s failed: %s", name, exc)
            path = None
        if path is not None:
            created[name] = str(path)
    return created
