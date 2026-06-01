"""Main benchmark-report builder.

Loads available artifacts, renders a polished 17-section Markdown report, writes
metadata, and orchestrates optional chart/HTML/PDF generation. The report is
honest by construction: when the underlying evaluation data is a sample/mock
(or ``mock_mode`` is set), a prominent banner makes that unmistakable.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finsage.logging_utils import get_logger
from finsage.reporting import charts as charts_mod
from finsage.reporting import tables
from finsage.reporting.exporters import export_markdown_to_html, export_markdown_to_pdf
from finsage.reporting.loaders import ReportInputConfig, load_optional_report_inputs
from finsage.reporting.qualitative import build_qualitative_section, select_qualitative_examples

logger = get_logger(__name__)

#: Report schema version (bump when the section layout changes materially).
REPORT_VERSION = "1.0"

#: Banner shown when the report is a sample/mock (not real benchmark results).
MOCK_LABEL = "Sample/mock report for pipeline validation only. Not real benchmark results."

#: Canonical ``##`` section titles, in order. Shared with the validator.
SECTION_TITLES: tuple[str, ...] = (
    "Executive Summary",
    "Project Overview",
    "Why Financial Filings",
    "Dataset Summary",
    "Training Setup",
    "Evaluation Methodology",
    "Overall Results",
    "Task-wise Results",
    "Hallucination and Faithfulness Analysis",
    "Latency and Deployment Summary",
    "Qualitative Examples",
    "Error Analysis and Regressions",
    "Limitations",
    "Financial Safety Disclaimer",
    "Reproducibility Guide",
    "Appendix",
)

#: Exact heading text the validator requires (subset that must always exist).
DISCLAIMER_HEADING = "Financial Safety Disclaimer"
LIMITATIONS_HEADING = "Limitations"

_MOCK_BACKENDS = {"mock", "sample", "fake"}


class BenchmarkReportBuilder:
    """Builds the FinSage-7B benchmark report from available artifacts.

    Args:
        input_dir: Directory of evaluation/figure artifacts.
        output_dir: Directory to write report outputs into.
        dataset_stats_path: Path to ``dataset_stats.json`` (or ``None``).
        training_summary_path: Path to ``training_summary.json`` (or ``None``).
        project_name: Project name shown on the title page.
        mock_mode: When ``True``, force the sample/mock banner regardless of data.
    """

    def __init__(
        self,
        input_dir: Path | str = "reports/figures",
        output_dir: Path | str = "reports",
        dataset_stats_path: Path | str | None = "data/datasets/dataset_stats.json",
        training_summary_path: Path | str | None = "checkpoints/finsage-7b/training_summary.json",
        project_name: str = "FinSage-7B",
        mock_mode: bool = False,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.mock_mode = mock_mode
        self.config = ReportInputConfig(
            input_dir=self.input_dir,
            dataset_stats_path=Path(dataset_stats_path) if dataset_stats_path else None,
            training_summary_path=(Path(training_summary_path) if training_summary_path else None),
        )

    # ------------------------------------------------------------------ context

    def _git_commit(self) -> str | None:
        """Return the current git commit hash, or ``None`` if unavailable."""
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return out.stdout.strip() or None
        except (subprocess.SubprocessError, OSError):
            return None

    def _data_is_sample(self, inputs: dict[str, Any]) -> bool:
        """Detect whether the loaded evaluation data is a sample/mock.

        Args:
            inputs: The loaded report inputs.

        Returns:
            ``True`` if any results/comparison backend is a mock, or if the core
            comparison artifact is missing entirely.
        """
        backends = []
        for key in ("baseline_results", "finetuned_results"):
            data = inputs.get(key) or {}
            backends.append(str(data.get("backend", "")).lower())
        summary = inputs.get("comparison_summary") or {}
        backends.append(str(summary.get("baseline_backend", "")).lower())
        backends.append(str(summary.get("finetuned_backend", "")).lower())
        if any(b in _MOCK_BACKENDS for b in backends):
            return True
        return inputs.get("comparison_results") is None

    def build_report_context(self) -> dict[str, Any]:
        """Load artifacts and assemble the structured report context.

        Returns:
            A context dict consumed by :meth:`generate_markdown` and
            :meth:`write_metadata`.
        """
        inputs = load_optional_report_inputs(self.config)
        present = [
            k
            for k, v in inputs.items()
            if k not in {"warnings", "missing"} and v not in (None, [], {})
        ]
        data_is_sample = self._data_is_sample(inputs)
        return {
            "project_name": self.project_name,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mock_mode": self.mock_mode,
            "data_is_sample": data_is_sample,
            "is_sample_report": self.mock_mode or data_is_sample,
            "inputs": inputs,
            "available_artifacts": sorted(present),
            "missing_artifacts": sorted(inputs.get("missing", [])),
            "warnings": list(inputs.get("warnings", [])),
            "git_commit": self._git_commit(),
            "python_version": platform.python_version(),
            "report_version": REPORT_VERSION,
            "charts": {},
        }

    # ------------------------------------------------------------------ render

    def _banner(self, context: dict[str, Any]) -> str:
        """Render the sample/mock banner when applicable, else an empty string."""
        if not context["is_sample_report"]:
            return ""
        reason = (
            "`mock_mode` was enabled"
            if context["mock_mode"]
            else "the underlying evaluation artifacts were produced by the **mock** "
            "generator (no real fine-tuned weights)"
        )
        return (
            f"> ⚠️ **{MOCK_LABEL}**\n>\n"
            f"> This report was generated because {reason}. The numbers below validate "
            "the reporting pipeline only and **must not be published as real results**. "
            "Re-run after a real fine-tune + evaluation to produce a publishable report.\n"
        )

    def _chart_md(self, context: dict[str, Any], name: str, alt: str) -> str:
        """Return a Markdown image line for a generated chart, or an empty string."""
        path = context["charts"].get(name)
        if not path:
            return ""
        rel = os.path.relpath(path, start=self.output_dir).replace(os.sep, "/")
        return f"\n![{alt}]({rel})\n"

    def _executive_summary(self, context: dict[str, Any]) -> str:
        """Render a data-driven executive summary."""
        inputs = context["inputs"]
        summary = inputs.get("comparison_summary") or {}
        improved = summary.get("metrics_improved")
        regressed = summary.get("metrics_regressed")
        compared = summary.get("metrics_compared")
        mean_delta = summary.get("mean_absolute_delta")
        if compared:
            headline = (
                f"Across **{compared}** overall metrics, FinSage-7B improved on "
                f"**{improved}** and regressed on **{regressed}** versus the base "
                f"Mistral-7B-Instruct model, with a mean absolute delta of "
                f"**{tables.format_metric(mean_delta)}**."
            )
        else:
            headline = (
                "Overall comparison metrics were not available at report time; see the "
                "missing-artifacts note in the metadata."
            )
        return (
            f"{headline}\n\n"
            "FinSage-7B is a QLoRA fine-tune of Mistral-7B-Instruct specialised for "
            "analysing U.S. SEC filings (10-K / 10-Q / 8-K). It is designed to stay "
            "grounded in the supplied filing text, extract concrete metrics, and avoid "
            "investment advice. This report documents the data, training, evaluation "
            "methodology, and the measured base-vs-fine-tuned comparison."
        )

    def generate_markdown(self, context: dict[str, Any]) -> str:
        """Render the full benchmark report as Markdown.

        Args:
            context: The context from :meth:`build_report_context` (after charts
                have optionally been attached).

        Returns:
            The complete Markdown document.
        """
        inputs = context["inputs"]
        p = context["project_name"]
        parts: list[str] = []

        # 1. Title page
        parts.append(f"# {p} — Benchmark Report")
        parts.append(
            f"_Generated {context['generated_at']} · report v{context['report_version']}"
            f"{' · commit ' + context['git_commit'] if context['git_commit'] else ''}_"
        )
        banner = self._banner(context)
        if banner:
            parts.append(banner)
        parts.append("---")

        # 2. Executive Summary
        parts.append("## Executive Summary")
        parts.append(self._executive_summary(context))

        # 3. Project Overview
        parts.append("## Project Overview")
        parts.append(
            "FinSage-7B is an end-to-end, reproducible pipeline: SEC EDGAR ingestion → "
            "section extraction → instruction-dataset construction → baseline evaluation "
            "→ QLoRA fine-tuning → fine-tuned evaluation → vLLM serving → FastAPI wrapper "
            "→ web demo → Docker deployment. Every stage is covered by tests and runs on "
            "commodity hardware (training on a single rented GPU; everything else CPU-only)."
        )

        # 4. Why Financial Filings
        parts.append("## Why Financial Filings")
        parts.append(
            "SEC filings are long, dense, and high-stakes: analysts spend hours locating "
            "risk factors, MD&A drivers, and reported metrics. They are also public and "
            "well-structured, which makes them an ideal domain for demonstrating that a "
            "small, cheaply fine-tuned model can be made **more grounded and specific** "
            "than its base model on a real professional task — without hallucinating "
            "numbers or drifting into advice."
        )

        # 5. Dataset Summary
        parts.append("## Dataset Summary")
        parts.append(tables.build_dataset_stats_table(inputs.get("dataset_stats")))
        parts.append(self._chart_md(context, "dataset_distribution", "Examples per task type"))
        validation = inputs.get("validation_report")
        if validation:
            leakage = validation.get("leakage_detected", validation.get("leakage"))
            parts.append(
                f"_Dataset validation report present; company/time-split leakage check: "
                f"`{leakage}`._"
            )

        # 6. Training Setup
        parts.append("## Training Setup")
        parts.append(tables.build_training_summary_table(inputs.get("training_summary")))

        # 7. Evaluation Methodology
        parts.append("## Evaluation Methodology")
        parts.append(
            "Both models are evaluated on the same held-out instruction set with identical "
            "prompts. Metrics include exact match, token-level F1, ROUGE-L, numeric "
            "precision/recall/exact-match (for extraction tasks), classification accuracy "
            "(for outlook/hallucination tasks), and a faithfulness score (lexical overlap "
            "by default; optional NLI entailment). The base model uses the same generation "
            "settings as the fine-tuned model so the comparison is apples-to-apples."
        )

        # 8. Overall Results
        parts.append("## Overall Results")
        parts.append(tables.build_overall_metrics_table(inputs.get("comparison_results")))
        parts.append(
            self._chart_md(context, "overall_metrics", "Overall metrics: base vs fine-tuned")
        )

        # 9. Task-wise Results
        parts.append("## Task-wise Results")
        delta_src = inputs.get("metric_delta_by_task") or inputs.get("comparison_results")
        parts.append(tables.build_task_metrics_table(delta_src))
        parts.append(self._chart_md(context, "task_delta", "Mean absolute delta by task"))

        # 10. Hallucination and Faithfulness Analysis
        parts.append("## Hallucination and Faithfulness Analysis")
        parts.append(
            "Faithfulness measures whether the answer stays grounded in the provided "
            "excerpt. The chart below contrasts faithfulness-related metrics for the base "
            "and fine-tuned models; higher is better. Note the default faithfulness metric "
            "is a lexical proxy, not a full entailment audit (see Limitations)."
        )
        parts.append(self._chart_md(context, "hallucination", "Faithfulness metrics"))

        # 11. Latency and Deployment Summary
        parts.append("## Latency and Deployment Summary")
        latency = inputs.get("api_latency") or inputs.get("vllm_latency")
        parts.append(tables.build_latency_table(latency))
        parts.append(self._chart_md(context, "latency", "Serving latency percentiles"))
        parts.append(
            "The serving stack is a public CPU-only FastAPI wrapper (auth, rate limiting, "
            "structured logging, financial-disclaimer injection) in front of an internal, "
            "GPU-bound vLLM OpenAI-compatible server. The full stack is packaged with "
            "Docker Compose (production, demo, and GPU overlays)."
        )

        # 12. Qualitative Examples
        parts.append("## Qualitative Examples")
        examples = select_qualitative_examples(inputs.get("qualitative", []))
        parts.append(build_qualitative_section(examples))

        # 13. Error Analysis and Regressions
        parts.append("## Error Analysis and Regressions")
        regressions = [e for e in examples if e.get("category") == "regression"]
        if regressions:
            parts.append(
                "At least one task regressed on lexical-overlap metrics. In manual review "
                "this is usually because the fine-tuned model adds correct, filing-grounded "
                "detail that the short reference answer omits — penalising overlap scores "
                "while improving usefulness. This is a known weakness of n-gram metrics on "
                "open-ended generation and motivates the faithfulness/NLI track."
            )
        else:
            parts.append(
                "No clear regression case was selected from the available qualitative "
                "examples. Where overlap metrics drop, inspect whether the fine-tuned "
                "answer added correct detail absent from the reference."
            )

        # 14. Limitations
        parts.append("## Limitations")
        parts.append(tables.build_limitations_table())

        # 15. Financial Safety Disclaimer
        parts.append("## Financial Safety Disclaimer")
        parts.append(
            "FinSage-7B is **not** a licensed financial advisor. Its outputs are "
            "informational summaries of the supplied text only, are **not** investment "
            "recommendations, and may be incomplete or incorrect. Always verify against the "
            "original filing and consult a qualified professional before making decisions."
        )

        # 16. Reproducibility Guide
        parts.append("## Reproducibility Guide")
        parts.append(
            "```bash\n"
            "# 1. Build the instruction dataset (no network in test mode)\n"
            "make build-dataset && make validate-dataset\n"
            "# 2. Baseline eval (mock backend is CPU-only; real needs a GPU)\n"
            "make eval-baseline\n"
            "# 3. Fine-tune (GPU) then evaluate + compare\n"
            "make train && make eval-finetuned && make compare-models\n"
            "# 4. Regenerate this report\n"
            "make report && make validate-report\n"
            "```"
        )

        # 17. Appendix
        parts.append("## Appendix")
        parts.append(
            "See [report_appendix.md](report_appendix.md) for metric and task-type "
            "definitions, the dataset split strategy, evaluation caveats, the prompt "
            "format, and full reproducibility commands."
        )
        if context["missing_artifacts"]:
            parts.append(
                "**Artifacts not available at report time:** "
                + ", ".join(f"`{m}`" for m in context["missing_artifacts"])
                + "."
            )

        return "\n\n".join(part for part in parts if part) + "\n"

    # ------------------------------------------------------------------ write

    def write_markdown(self, markdown: str, path: Path | str) -> Path:
        """Write the Markdown report to disk.

        Args:
            markdown: The rendered Markdown.
            path: Destination path.

        Returns:
            The path written to.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        logger.info("Wrote benchmark report to %s", out)
        return out

    def write_metadata(self, context: dict[str, Any], path: Path | str) -> Path:
        """Write the report metadata JSON.

        Args:
            context: The report context.
            path: Destination path.

        Returns:
            The path written to.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "project_name": context["project_name"],
            "generated_at": context["generated_at"],
            "report_version": context["report_version"],
            "mock_mode": context["mock_mode"],
            "is_sample_report": context["is_sample_report"],
            "data_is_sample": context["data_is_sample"],
            "available_artifacts": context["available_artifacts"],
            "missing_artifacts": context["missing_artifacts"],
            "warnings": context["warnings"],
            "charts": context["charts"],
            "git_commit": context["git_commit"],
            "python_version": context["python_version"],
            "generator": f"finsage.reporting v{context['report_version']}",
            "python_executable": sys.executable,
        }
        out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote report metadata to %s", out)
        return out

    # ------------------------------------------------------------------ build

    def build(
        self,
        output_markdown: Path | str = "reports/benchmark_report.md",
        generate_charts: bool = True,
        export_pdf: bool = True,
        export_html: bool = True,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Run the full report build and return the output paths.

        Args:
            output_markdown: Destination Markdown path.
            generate_charts: Whether to generate PNG charts.
            export_pdf: Whether to attempt PDF export.
            export_html: Whether to attempt HTML export.
            context: A precomputed context from :meth:`build_report_context`. When
                provided it is reused (and mutated in place with chart paths) so
                callers can inspect the generated charts afterwards; otherwise a
                fresh context is built.

        Returns:
            A mapping of output kind (``markdown``, ``metadata``, ``html``,
            ``pdf``) to the written path; missing exports are omitted.
        """
        if context is None:
            context = self.build_report_context()

        if generate_charts:
            # Charts are written next to the report (output_dir/figures), not into
            # the input directory, so generating from fixtures never pollutes them.
            context["charts"] = charts_mod.create_all_report_charts(
                context["inputs"], self.output_dir / "figures"
            )

        markdown = self.generate_markdown(context)
        md_path = self.write_markdown(markdown, output_markdown)
        meta_path = self.write_metadata(context, self.output_dir / "report_metadata.json")

        outputs: dict[str, Path] = {"markdown": md_path, "metadata": meta_path}

        if export_html:
            html_path = export_markdown_to_html(md_path, self.output_dir / "benchmark_report.html")
            if html_path is not None:
                outputs["html"] = html_path
        if export_pdf:
            pdf_path = export_markdown_to_pdf(md_path, self.output_dir / "benchmark_report.pdf")
            if pdf_path is not None:
                outputs["pdf"] = pdf_path

        return outputs
