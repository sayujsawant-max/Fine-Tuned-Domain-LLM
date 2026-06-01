"""Standalone CLI to compare already-generated baseline/fine-tuned outputs (Phase 6).

Use this when both evaluations have already been run and you just want the
comparison artifacts, charts, and benchmark report.

Example::

    python evaluation/compare_models.py \\
        --baseline-results reports/figures/baseline_results.json \\
        --baseline-predictions reports/figures/baseline_predictions.jsonl \\
        --finetuned-results reports/figures/finetuned_results.json \\
        --finetuned-predictions reports/figures/finetuned_predictions.jsonl \\
        --output-dir reports/figures --report-path reports/benchmark_report.md
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from finsage.config import get_settings
from finsage.evaluation import visualization
from finsage.evaluation.comparison import ModelComparison
from finsage.evaluation.report_generator import BenchmarkReportGenerator
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Compare baseline vs fine-tuned evaluation outputs.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.command()
def main(
    baseline_results: str = typer.Option(..., "--baseline-results", help="Baseline results JSON."),
    baseline_predictions: str = typer.Option(
        ..., "--baseline-predictions", help="Baseline predictions JSONL."
    ),
    finetuned_results: str = typer.Option(
        ..., "--finetuned-results", help="Fine-tuned results JSON."
    ),
    finetuned_predictions: str = typer.Option(
        ..., "--finetuned-predictions", help="Fine-tuned predictions JSONL."
    ),
    output_dir: str = typer.Option("reports/figures", help="Comparison output directory."),
    report_path: str = typer.Option("reports/benchmark_report.md", help="Report output path."),
    generate_charts: bool = typer.Option(
        True, "--generate-charts/--no-generate-charts", help="Render PNG charts."
    ),
) -> None:
    """Compare existing evaluation outputs and write the benchmark report.

    Args:
        baseline_results: Path to the baseline results JSON.
        baseline_predictions: Path to the baseline predictions JSONL.
        finetuned_results: Path to the fine-tuned results JSON.
        finetuned_predictions: Path to the fine-tuned predictions JSONL.
        output_dir: Comparison output directory.
        report_path: Markdown report output path.
        generate_charts: Whether to render charts.

    Raises:
        typer.Exit: With code 1 if any input file is missing.
    """
    setup_logging(get_settings().log_level)

    comparator = ModelComparison()
    try:
        base_results = comparator.load_json(baseline_results)
        fine_results = comparator.load_json(finetuned_results)
        base_preds = comparator.load_jsonl(baseline_predictions)
        fine_preds = comparator.load_jsonl(finetuned_predictions)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    comparison = comparator.compare_results(base_results, fine_results)
    qualitative = comparator.compare_predictions(base_preds, fine_preds, max_examples=10)
    comparator.write_comparison_outputs(comparison, qualitative, output_dir)

    chart_paths: dict[str, str] = {}
    if generate_charts:
        out = Path(output_dir)
        for name, path in (
            (
                "overall_metric_comparison",
                visualization.plot_overall_metric_comparison(
                    comparison, out / "overall_metric_comparison.png"
                ),
            ),
            (
                "task_metric_deltas",
                visualization.plot_task_metric_deltas(comparison, out / "task_metric_deltas.png"),
            ),
            (
                "task_distribution",
                visualization.plot_task_counts(fine_results, out / "task_distribution.png"),
            ),
        ):
            if path is not None:
                chart_paths[name] = f"figures/{Path(path).name}"

    report = BenchmarkReportGenerator().generate_benchmark_report(
        base_results, fine_results, comparison, qualitative, report_path, chart_paths or None
    )

    summary = comparison.get("summary", {})
    console.print(
        f"[green]Comparison complete.[/green] Improved "
        f"{summary.get('metrics_improved', 0)}/{summary.get('metrics_compared', 0)} metrics."
    )
    console.print(f"Report: {report}")


if __name__ == "__main__":
    app()
