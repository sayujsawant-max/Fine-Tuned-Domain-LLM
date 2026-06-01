"""Compare baseline vs fine-tuned evaluation outputs.

Computes per-metric deltas (overall and by task), finds the biggest improvements
and regressions, and joins predictions by example id for qualitative review. All
outputs are JSON-serialisable and free of NaN/Infinity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


def _safe_float(value: Any) -> float:
    """Coerce a value to a finite float (non-finite -> 0.0).

    Args:
        value: Any value.

    Returns:
        A finite float; ``0.0`` for ``None``/non-numeric/NaN/Inf.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _delta(baseline: float, finetuned: float) -> dict[str, Any]:
    """Build a delta record for one metric.

    Args:
        baseline: Baseline metric value.
        finetuned: Fine-tuned metric value.

    Returns:
        A mapping with baseline/finetuned values, absolute and relative deltas,
        and an ``improved`` flag.
    """
    abs_delta = round(finetuned - baseline, 4)
    rel = (abs_delta / abs(baseline) * 100.0) if baseline != 0 else (100.0 if abs_delta else 0.0)
    return {
        "baseline": round(baseline, 4),
        "finetuned": round(finetuned, 4),
        "absolute_delta": abs_delta,
        "relative_delta_pct": round(rel, 2),
        "improved": abs_delta > 0,
    }


class ModelComparison:
    """Compares baseline and fine-tuned evaluation artifacts."""

    @staticmethod
    def load_json(path: Path | str) -> dict[str, Any]:
        """Load a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            The parsed mapping.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return dict(json.loads(p.read_text(encoding="utf-8")))

    @staticmethod
    def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
        """Load a JSONL file into a list of dicts.

        Args:
            path: Path to the JSONL file.

        Returns:
            The parsed rows.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        rows: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def compare_results(
        self,
        baseline_results: dict[str, Any],
        finetuned_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare baseline and fine-tuned aggregate results.

        Args:
            baseline_results: Baseline results dict (from ``EvalRunner``).
            finetuned_results: Fine-tuned results dict.

        Returns:
            A dict with ``overall_comparison``, ``by_task_comparison``,
            ``improvements``, ``regressions``, ``warnings``, and ``summary``.
        """
        base_overall = baseline_results.get("overall", {})
        fine_overall = finetuned_results.get("overall", {})
        warnings: list[str] = []

        overall_comparison = self._compare_metric_dict(
            base_overall, fine_overall, warnings, "overall"
        )

        base_by_task = baseline_results.get("by_task", {})
        fine_by_task = finetuned_results.get("by_task", {})
        by_task_comparison: dict[str, Any] = {}
        for task in sorted(set(base_by_task) | set(fine_by_task)):
            by_task_comparison[task] = self._compare_metric_dict(
                base_by_task.get(task, {}), fine_by_task.get(task, {}), warnings, task
            )

        improvements = {m: d for m, d in overall_comparison.items() if d["absolute_delta"] > 0}
        regressions = {m: d for m, d in overall_comparison.items() if d["absolute_delta"] < 0}

        summary = {
            "metrics_compared": len(overall_comparison),
            "metrics_improved": len(improvements),
            "metrics_regressed": len(regressions),
            "mean_absolute_delta": round(
                sum(d["absolute_delta"] for d in overall_comparison.values())
                / max(1, len(overall_comparison)),
                4,
            ),
            "baseline_backend": baseline_results.get("backend"),
            "finetuned_backend": finetuned_results.get("backend"),
            "num_examples_baseline": baseline_results.get("num_examples"),
            "num_examples_finetuned": finetuned_results.get("num_examples"),
        }
        return {
            "overall_comparison": overall_comparison,
            "by_task_comparison": by_task_comparison,
            "improvements": improvements,
            "regressions": regressions,
            "warnings": warnings,
            "summary": summary,
        }

    @staticmethod
    def _compare_metric_dict(
        baseline: dict[str, Any],
        finetuned: dict[str, Any],
        warnings: list[str],
        scope: str,
    ) -> dict[str, Any]:
        """Compare two metric mappings, noting metrics missing on either side."""
        comparison: dict[str, Any] = {}
        for metric in sorted(set(baseline) | set(finetuned)):
            if metric not in baseline:
                warnings.append(f"[{scope}] metric '{metric}' missing in baseline")
            if metric not in finetuned:
                warnings.append(f"[{scope}] metric '{metric}' missing in finetuned")
            comparison[metric] = _delta(
                _safe_float(baseline.get(metric)), _safe_float(finetuned.get(metric))
            )
        return comparison

    def compare_predictions(
        self,
        baseline_predictions: list[dict[str, Any]],
        finetuned_predictions: list[dict[str, Any]],
        max_examples: int = 10,
    ) -> list[dict[str, Any]]:
        """Join baseline and fine-tuned predictions by id for qualitative review.

        Args:
            baseline_predictions: Baseline prediction rows.
            finetuned_predictions: Fine-tuned prediction rows.
            max_examples: Maximum number of joined rows to return.

        Returns:
            Qualitative comparison rows joined on ``id``.
        """
        fine_by_id = {row.get("id"): row for row in finetuned_predictions}
        rows: list[dict[str, Any]] = []
        for base in baseline_predictions:
            fine = fine_by_id.get(base.get("id"))
            if fine is None:
                continue
            base_metrics = base.get("metrics", {})
            fine_metrics = fine.get("metrics", {})
            rows.append(
                {
                    "id": base.get("id"),
                    "task_type": base.get("task_type"),
                    "instruction": base.get("instruction"),
                    "input_preview": base.get("input_preview"),
                    "reference": base.get("reference"),
                    "baseline_prediction": base.get("prediction"),
                    "finetuned_prediction": fine.get("prediction"),
                    "baseline_metrics": base_metrics,
                    "finetuned_metrics": fine_metrics,
                    "improvement_summary": self._improvement_summary(base_metrics, fine_metrics),
                }
            )
            if len(rows) >= max_examples:
                break
        return rows

    @staticmethod
    def _improvement_summary(
        baseline_metrics: dict[str, Any], finetuned_metrics: dict[str, Any]
    ) -> dict[str, float]:
        """Return per-metric absolute deltas for a single prediction pair."""
        summary: dict[str, float] = {}
        for metric in sorted(set(baseline_metrics) | set(finetuned_metrics)):
            summary[metric] = round(
                _safe_float(finetuned_metrics.get(metric))
                - _safe_float(baseline_metrics.get(metric)),
                4,
            )
        return summary

    @staticmethod
    def _rank(
        comparison_rows: list[dict[str, Any]], metric: str, limit: int, reverse: bool
    ) -> list[dict[str, Any]]:
        """Rank comparison rows by the delta of a given metric."""
        scored = [row for row in comparison_rows if metric in row.get("improvement_summary", {})]
        scored.sort(key=lambda r: r["improvement_summary"][metric], reverse=reverse)
        return scored[:limit]

    def find_best_improvements(
        self,
        comparison_rows: list[dict[str, Any]],
        metric: str = "token_f1",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the rows with the largest positive delta for ``metric``.

        Args:
            comparison_rows: Rows from :meth:`compare_predictions`.
            metric: The metric to rank by.
            limit: Maximum number of rows to return.

        Returns:
            The top-improving rows (delta descending).
        """
        return self._rank(comparison_rows, metric, limit, reverse=True)

    def find_regressions(
        self,
        comparison_rows: list[dict[str, Any]],
        metric: str = "token_f1",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the rows with the largest negative delta for ``metric``.

        Args:
            comparison_rows: Rows from :meth:`compare_predictions`.
            metric: The metric to rank by.
            limit: Maximum number of rows to return.

        Returns:
            The most-regressed rows (delta ascending), excluding non-negative ones.
        """
        ranked = self._rank(comparison_rows, metric, limit, reverse=False)
        return [row for row in ranked if row["improvement_summary"][metric] < 0]

    def write_comparison_outputs(
        self,
        comparison: dict[str, Any],
        qualitative_rows: list[dict[str, Any]],
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """Write the comparison artifacts to disk.

        Args:
            comparison: The comparison dict from :meth:`compare_results`.
            qualitative_rows: Rows from :meth:`compare_predictions`.
            output_dir: Destination directory.

        Returns:
            A mapping of artifact name to written path.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _write_json(name: str, payload: Any) -> Path:
            path = out_dir / name
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            return path

        paths = {
            "comparison_results": _write_json("comparison_results.json", comparison),
            "metric_delta_by_task": _write_json(
                "metric_delta_by_task.json", comparison.get("by_task_comparison", {})
            ),
            "comparison_summary": _write_json(
                "comparison_summary.json", comparison.get("summary", {})
            ),
        }
        qual_path = out_dir / "qualitative_comparisons.jsonl"
        with qual_path.open("w", encoding="utf-8") as fh:
            for row in qualitative_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths["qualitative_comparisons"] = qual_path
        logger.info("Wrote comparison outputs to %s", out_dir)
        return paths
