"""Lightweight chart generation for the benchmark report (matplotlib only).

matplotlib is imported lazily and every function degrades gracefully: if
matplotlib is unavailable or rendering fails, a warning is logged and ``None`` is
returned — chart generation never fails the evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


def _import_pyplot() -> Any:
    """Import ``matplotlib.pyplot`` with a non-interactive backend, or ``None``.

    Returns:
        The ``pyplot`` module, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - any import/backend error means skip charts
        logger.warning("matplotlib unavailable; skipping chart generation.")
        return None
    return plt


def plot_overall_metric_comparison(
    comparison: dict[str, Any], output_path: Path | str
) -> Path | None:
    """Plot baseline vs fine-tuned values for each overall metric.

    Args:
        comparison: A comparison dict (see ``ModelComparison.compare_results``).
        output_path: Destination PNG path.

    Returns:
        The written path, or ``None`` if charting was skipped/failed.
    """
    plt = _import_pyplot()
    if plt is None:
        return None
    try:
        overall = comparison.get("overall_comparison", {})
        if not overall:
            return None
        metrics = list(overall)
        baseline = [overall[m]["baseline"] for m in metrics]
        finetuned = [overall[m]["finetuned"] for m in metrics]
        x = range(len(metrics))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.2), 4))
        ax.bar([i - width / 2 for i in x], baseline, width, label="baseline")
        ax.bar([i + width / 2 for i in x], finetuned, width, label="finetuned")
        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics, rotation=30, ha="right")
        ax.set_ylabel("score")
        ax.set_title("Overall metrics: base vs fine-tuned")
        ax.legend()
        fig.tight_layout()
        return _save(fig, plt, output_path)
    except Exception:  # noqa: BLE001 - never fail evaluation on a chart
        logger.warning("Failed to render overall metric comparison chart.")
        return None


def plot_task_metric_deltas(
    comparison: dict[str, Any], output_path: Path | str, metric: str = "token_f1"
) -> Path | None:
    """Plot the per-task absolute delta for a metric.

    Args:
        comparison: A comparison dict.
        output_path: Destination PNG path.
        metric: The metric whose per-task delta to plot.

    Returns:
        The written path, or ``None`` if charting was skipped/failed.
    """
    plt = _import_pyplot()
    if plt is None:
        return None
    try:
        by_task = comparison.get("by_task_comparison", {})
        tasks = [t for t in sorted(by_task) if metric in by_task[t]]
        if not tasks:
            return None
        deltas = [by_task[t][metric]["absolute_delta"] for t in tasks]

        fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 1.0), 4))
        ax.bar(tasks, deltas, color=["#2a9d8f" if d >= 0 else "#e76f51" for d in deltas])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticklabels(tasks, rotation=40, ha="right")
        ax.set_ylabel(f"Δ {metric}")
        ax.set_title(f"Per-task {metric} delta (fine-tuned − base)")
        fig.tight_layout()
        return _save(fig, plt, output_path)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to render task metric delta chart.")
        return None


def plot_task_counts(results: dict[str, Any], output_path: Path | str) -> Path | None:
    """Plot the number of examples per task type.

    Args:
        results: A results dict containing ``count_by_task``.
        output_path: Destination PNG path.

    Returns:
        The written path, or ``None`` if charting was skipped/failed.
    """
    plt = _import_pyplot()
    if plt is None:
        return None
    try:
        counts = results.get("count_by_task", {})
        if not counts:
            return None
        tasks = list(counts)
        values = [counts[t] for t in tasks]

        fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 1.0), 4))
        ax.bar(tasks, values, color="#264653")
        ax.set_xticklabels(tasks, rotation=40, ha="right")
        ax.set_ylabel("examples")
        ax.set_title("Test-set task distribution")
        fig.tight_layout()
        return _save(fig, plt, output_path)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to render task distribution chart.")
        return None


def _save(fig: Any, plt: Any, output_path: Path | str) -> Path:
    """Save a figure to disk and close it.

    Args:
        fig: The matplotlib figure.
        plt: The pyplot module.
        output_path: Destination PNG path.

    Returns:
        The written path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved chart to %s", out_path)
    return out_path
