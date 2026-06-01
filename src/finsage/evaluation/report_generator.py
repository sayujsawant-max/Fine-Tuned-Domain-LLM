"""Render evaluation results into Markdown reports.

Two report styles live here:

- :func:`build_comparison_table` / :func:`render_report` — the base-vs-FinSage
  comparison table used in ``reports/benchmark_report.md`` (Phase 7).
- :class:`BaselineReportGenerator` — the Phase 4 baseline evaluation report.

Both are dependency-free and never fabricate metrics.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "FinSage-7B is not a licensed financial advisor. Outputs are not investment "
    "recommendations and may be inaccurate. Always verify against the original "
    "filings. This baseline uses the un-fine-tuned base model."
)


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


def _metric_table(metrics: dict[str, float]) -> str:
    """Render a metric mapping as a two-column Markdown table."""
    if not metrics:
        return "_No metrics available._"
    lines = ["| Metric | Value |", "|--------|-------|"]
    for name, value in sorted(metrics.items()):
        lines.append(f"| {name} | {float(value):.4f} |")
    return "\n".join(lines)


class BaselineReportGenerator:
    """Generates the Phase 4 baseline evaluation Markdown report."""

    project_name = "FinSage-7B"

    def generate_markdown_report(
        self,
        results: dict[str, Any],
        predictions_path: Path | str,
        output_path: Path | str = "reports/baseline_eval_report.md",
    ) -> Path:
        """Render and write the baseline evaluation report.

        Args:
            results: The aggregated results dict from :class:`EvalRunner`.
            predictions_path: Path to the predictions JSONL (for qualitative
                examples and provenance).
            output_path: Destination Markdown path.

        Returns:
            The path written to.
        """
        overall = results.get("overall", {})
        by_task = results.get("by_task", {})
        count_by_task = results.get("count_by_task", {})

        sections: list[str] = [
            f"# {self.project_name} — Baseline Evaluation Report",
            "",
            f"- **Evaluation date:** {date.today().isoformat()}",
            f"- **Backend:** {results.get('backend', 'unknown')}",
            f"- **Model ID:** {results.get('model_id') or 'n/a (mock backend)'}",
            f"- **Test file:** {results.get('test_file', 'n/a')}",
            f"- **Examples evaluated:** {results.get('num_examples', 0)}",
            f"- **Avg input length (chars):** {results.get('average_input_length', 0)}",
            f"- **Avg prediction length (chars):** {results.get('average_prediction_length', 0)}",
            "",
            "## Overall metrics",
            "",
            _metric_table(overall),
            "",
            "## Metrics by task",
            "",
        ]

        for task in sorted(by_task):
            sections.append(f"### {task} (n={count_by_task.get(task, 0)})")
            sections.append("")
            sections.append(_metric_table(by_task[task]))
            sections.append("")

        sections.append("## Task distribution")
        sections.append("")
        sections.append("| Task type | Examples |")
        sections.append("|-----------|----------|")
        for task, count in sorted(count_by_task.items()):
            sections.append(f"| {task} | {count} |")
        sections.append("")

        sections.append("## Qualitative examples")
        sections.append("")
        for example in self._load_examples(predictions_path, limit=3):
            sections.append(f"- **Task:** {example.get('task_type', '')}  ")
            sections.append(f"  **Instruction:** {example.get('instruction', '')}  ")
            sections.append(f"  **Reference:** {self._truncate(example.get('reference', ''))}  ")
            sections.append(f"  **Prediction:** {self._truncate(example.get('prediction', ''))}")
            sections.append("")

        sections.append("## Limitations")
        sections.append("")
        sections.append(
            "- This is the **base model before fine-tuning**; results are a "
            "reference point, not a target.\n"
            "- Reference targets are template/extractive weak supervision "
            "(Phase 3), so absolute scores should be read with caution.\n"
            "- `lexical_faithfulness` is a lexical proxy, not a true NLI metric.\n"
            "- Mock-backend reports exist only to validate the pipeline."
        )
        sections.append("")
        sections.append("## Disclaimer")
        sections.append("")
        sections.append(f"> {DISCLAIMER}")
        sections.append("")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(sections), encoding="utf-8")
        logger.info("Wrote baseline report to %s", out_path)
        return out_path

    @staticmethod
    def _truncate(text: Any, limit: int = 200) -> str:
        """Truncate text to ``limit`` characters for the report."""
        s = str(text).replace("\n", " ")
        return s if len(s) <= limit else s[:limit] + "…"

    @staticmethod
    def _load_examples(predictions_path: Path | str, limit: int = 3) -> list[dict[str, Any]]:
        """Load up to ``limit`` prediction rows for qualitative display."""
        path = Path(predictions_path)
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
        return rows


def _comparison_table(overall_comparison: dict[str, Any]) -> str:
    """Render an overall base-vs-fine-tuned comparison table."""
    if not overall_comparison:
        return "_No overall comparison available._"
    lines = [
        "| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |",
        "|--------|------|------------|-------|-----|----------|",
    ]
    for metric, d in sorted(overall_comparison.items()):
        mark = "✅" if d["improved"] else ("➖" if d["absolute_delta"] == 0 else "🔻")
        lines.append(
            f"| {metric} | {d['baseline']:.4f} | {d['finetuned']:.4f} | "
            f"{d['absolute_delta']:+.4f} | {d['relative_delta_pct']:+.1f}% | {mark} |"
        )
    return "\n".join(lines)


class BenchmarkReportGenerator:
    """Generates the publishable before/after benchmark report (Markdown)."""

    project_name = "FinSage-7B"

    def generate_benchmark_report(
        self,
        baseline_results: dict[str, Any],
        finetuned_results: dict[str, Any],
        comparison: dict[str, Any],
        qualitative_examples: list[dict[str, Any]],
        output_path: Path | str = "reports/benchmark_report.md",
        chart_paths: dict[str, str] | None = None,
    ) -> Path:
        """Render and write the before/after benchmark report.

        Args:
            baseline_results: Baseline results dict.
            finetuned_results: Fine-tuned results dict.
            comparison: Comparison dict from ``ModelComparison.compare_results``.
            qualitative_examples: Joined qualitative rows.
            output_path: Destination Markdown path.
            chart_paths: Optional mapping of chart name -> image path to embed.

        Returns:
            The path written to.
        """
        summary = comparison.get("summary", {})
        is_mock = (
            baseline_results.get("backend") == "mock" or finetuned_results.get("backend") == "mock"
        )
        lines: list[str] = [f"# {self.project_name} Benchmark Report", ""]

        if is_mock:
            lines += [
                "> ⚠️ **Sample/mock numbers are for pipeline validation only and are "
                "not real benchmark results.** Run the adapter/merged backend on a GPU "
                "for real figures.",
                "",
            ]

        improved = summary.get("metrics_improved", 0)
        compared = summary.get("metrics_compared", 0)
        lines += [
            "## Executive summary",
            "",
            f"FinSage-7B (fine-tuned) improved **{improved}/{compared}** overall metrics "
            f"versus the base model, with a mean absolute delta of "
            f"**{summary.get('mean_absolute_delta', 0):+.4f}** across metrics.",
            "",
            "## Models",
            "",
            f"- **Base model:** {baseline_results.get('model_id') or 'mistralai/Mistral-7B-Instruct-v0.3'}",
            f"- **Fine-tuned backend:** {finetuned_results.get('backend')}",
            f"- **Fine-tuned model/adapter:** {finetuned_results.get('model_id') or 'see run config'}",
            "",
            "## Dataset",
            "",
            f"- **Test file:** {finetuned_results.get('test_file', 'n/a')}",
            f"- **Examples:** {finetuned_results.get('num_examples', 0)} "
            "(same held-out set as the baseline)",
            "",
            "## Task distribution",
            "",
            "| Task type | Examples |",
            "|-----------|----------|",
        ]
        for task, count in sorted(finetuned_results.get("count_by_task", {}).items()):
            lines.append(f"| {task} | {count} |")
        lines += [
            "",
            "## Overall metrics (base vs fine-tuned)",
            "",
            _comparison_table(comparison.get("overall_comparison", {})),
            "",
        ]

        lines += ["## Metrics by task", ""]
        for task, metrics in sorted(comparison.get("by_task_comparison", {}).items()):
            lines += [f"### {task}", "", _comparison_table(metrics), ""]

        lines += ["## Best improvements", ""]
        improvements = sorted(
            comparison.get("improvements", {}).items(),
            key=lambda kv: kv[1]["absolute_delta"],
            reverse=True,
        )
        if improvements:
            for metric, d in improvements:
                lines.append(
                    f"- **{metric}**: {d['baseline']:.4f} → {d['finetuned']:.4f} ({d['absolute_delta']:+.4f})"
                )
        else:
            lines.append("_No overall metric improved._")
        lines.append("")

        lines += ["## Regressions / failure cases", ""]
        regressions = comparison.get("regressions", {})
        if regressions:
            for metric, d in sorted(regressions.items(), key=lambda kv: kv[1]["absolute_delta"]):
                lines.append(
                    f"- **{metric}**: {d['baseline']:.4f} → {d['finetuned']:.4f} ({d['absolute_delta']:+.4f})"
                )
        else:
            lines.append("_No overall metric regressed._")
        lines.append("")

        lines += ["## Qualitative examples", ""]
        for ex in qualitative_examples[:5]:
            lines += [
                f"**Task:** {ex.get('task_type')} — {ex.get('id')}  ",
                f"**Instruction:** {self._truncate(ex.get('instruction', ''))}  ",
                f"**Reference:** {self._truncate(ex.get('reference', ''))}  ",
                f"**Base model:** {self._truncate(ex.get('baseline_prediction', ''))}  ",
                f"**FinSage-7B:** {self._truncate(ex.get('finetuned_prediction', ''))}  ",
                f"**Metric change:** {ex.get('improvement_summary', {})}",
                "",
            ]

        if chart_paths:
            lines += ["## Charts", ""]
            for name, rel in chart_paths.items():
                lines.append(f"![{name}]({rel})")
            lines.append("")

        lines += [
            "## Faithfulness / hallucination",
            "",
            "Faithfulness is currently approximated by `lexical_faithfulness` (lexical "
            "overlap with the source). A true NLI-based faithfulness metric and citation "
            "precision are planned (see eval guide).",
            "",
            "## Limitations",
            "",
            "- Phase 3 reference targets are template/extractive weak supervision.\n"
            "- `lexical_faithfulness` is a proxy, not entailment.\n"
            "- Small test sets and mock backends can mislead — read deltas with caution.",
            "",
            "## Disclaimer",
            "",
            f"> {DISCLAIMER}",
            "",
            "## Next steps",
            "",
            "- Run the real adapter/merged backend on a GPU for headline numbers.\n"
            "- Add NLI faithfulness, an LLM-as-judge rubric, and bootstrap confidence intervals.\n"
            "- Serve the merged model (Phase 7, vLLM).",
            "",
        ]

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote benchmark report to %s", out_path)
        return out_path

    @staticmethod
    def _truncate(text: Any, limit: int = 200) -> str:
        """Truncate text to ``limit`` characters for the report."""
        s = str(text).replace("\n", " ")
        return s if len(s) <= limit else s[:limit] + "…"

    @staticmethod
    def optionally_export_pdf(markdown_path: Path | str, pdf_path: Path | str) -> Path | None:
        """Export the Markdown report to PDF via pandoc, if available.

        Args:
            markdown_path: Path to the Markdown report.
            pdf_path: Destination PDF path.

        Returns:
            The PDF path if exported, else ``None`` (pandoc missing or failed).
        """
        import shutil
        import subprocess

        if shutil.which("pandoc") is None:
            logger.warning("pandoc not found; skipping PDF export.")
            return None
        try:
            subprocess.run(
                ["pandoc", str(markdown_path), "-o", str(pdf_path)],
                check=True,
                capture_output=True,
            )
        except (subprocess.SubprocessError, OSError):
            logger.warning("pandoc PDF export failed; skipping.")
            return None
        logger.info("Exported PDF to %s", pdf_path)
        return Path(pdf_path)
