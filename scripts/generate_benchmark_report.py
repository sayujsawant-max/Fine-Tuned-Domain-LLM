"""CLI to generate the FinSage-7B benchmark report (Phase 11).

Loads available evaluation/dataset/training/latency artifacts and renders a
polished Markdown report (with optional charts, HTML, and PDF). Missing optional
artifacts are tolerated in non-strict mode; ``--strict`` fails when core result
files are absent. No GPU, no model downloads, and no external services are used.

Example::

    python scripts/generate_benchmark_report.py generate \\
        --input-dir reports/figures \\
        --output-dir reports \\
        --generate-charts --export-pdf --export-html
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging
from finsage.reporting.report_builder import BenchmarkReportBuilder

app = typer.Typer(help="Generate the FinSage-7B benchmark report.", add_completion=False)
logger = get_logger(__name__)
console = Console()

#: Logical artifacts that must be present for a non-sample (strict) report.
_CORE_ARTIFACTS = ("baseline_results", "finetuned_results", "comparison_results")


@app.callback()
def _main() -> None:
    """Benchmark report generator (use the ``generate`` subcommand)."""


@app.command()
def generate(
    input_dir: str = typer.Option("reports/figures", help="Directory of evaluation artifacts."),
    output_dir: str = typer.Option("reports", help="Directory to write report outputs."),
    dataset_stats_path: str = typer.Option(
        "data/datasets/dataset_stats.json", help="Path to dataset_stats.json."
    ),
    training_summary_path: str = typer.Option(
        "checkpoints/finsage-7b/training_summary.json", help="Path to training_summary.json."
    ),
    generate_charts: bool = typer.Option(True, help="Generate PNG charts (needs matplotlib)."),
    export_pdf: bool = typer.Option(True, help="Attempt PDF export."),
    export_html: bool = typer.Option(True, help="Attempt HTML export."),
    mock_mode: bool = typer.Option(False, help="Force the sample/mock banner."),
    strict: bool = typer.Option(False, help="Fail if core result files are missing."),
) -> None:
    """Generate the benchmark report.

    Args:
        input_dir: Directory of evaluation/figure artifacts.
        output_dir: Directory to write the report outputs.
        dataset_stats_path: Path to ``dataset_stats.json``.
        training_summary_path: Path to ``training_summary.json``.
        generate_charts: Whether to generate charts.
        export_pdf: Whether to attempt PDF export.
        export_html: Whether to attempt HTML export.
        mock_mode: Force the sample/mock banner regardless of data.
        strict: Fail (exit 1) if any core result file is missing.

    Raises:
        typer.Exit: Code 1 when strict mode finds missing core artifacts.
    """
    setup_logging(get_settings().log_level)
    builder = BenchmarkReportBuilder(
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_stats_path=dataset_stats_path,
        training_summary_path=training_summary_path,
        mock_mode=mock_mode,
    )

    context = builder.build_report_context()
    missing_core = [a for a in _CORE_ARTIFACTS if a in context["missing_artifacts"]]
    if strict and missing_core:
        console.print(
            "[red]Strict mode: missing core artifacts: " + ", ".join(missing_core) + "[/red]"
        )
        raise typer.Exit(code=1)

    if context["is_sample_report"]:
        console.print(
            "[yellow]Note: producing a SAMPLE/MOCK report (not real benchmark results).[/yellow]"
        )

    outputs = builder.build(
        output_markdown=str(Path(output_dir) / "benchmark_report.md"),
        generate_charts=generate_charts,
        export_pdf=export_pdf,
        export_html=export_html,
        context=context,
    )

    table = Table(title="Benchmark report outputs", show_header=True, header_style="bold")
    table.add_column("Output")
    table.add_column("Path")
    for kind in ("markdown", "metadata", "html", "pdf"):
        table.add_row(kind, str(outputs.get(kind, "— skipped —")))
    table.add_row("charts", str(len(context["charts"])) + " generated")
    console.print(table)

    if "pdf" not in outputs and export_pdf:
        console.print("[yellow]PDF not produced; see reports/PDF_EXPORT_SKIPPED.md.[/yellow]")
    if context["warnings"]:
        console.print(
            f"[yellow]{len(context['warnings'])} warning(s) recorded in metadata.[/yellow]"
        )
    console.print("[green]Report generation complete.[/green]")


if __name__ == "__main__":
    app()
