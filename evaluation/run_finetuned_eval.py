"""CLI to evaluate the fine-tuned model and compare against the baseline (Phase 6).

Backends:

- ``mock``    — deterministic ``MockGenerator(quality="finetuned")`` (tests/CI).
- ``adapter`` — base model + LoRA adapter (PEFT); requires ``.[ml,training]`` + GPU.
- ``merged``  — already-merged fine-tuned model; requires ``.[ml,training]`` + GPU.

Example (mock)::

    python evaluation/run_finetuned_eval.py \\
        --test-file tests/fixtures/eval_test_sample.jsonl \\
        --baseline-results tests/fixtures/baseline_results_sample.json \\
        --baseline-predictions tests/fixtures/baseline_predictions_sample.jsonl \\
        --output-dir /tmp/finsage_finetuned_eval --backend mock --max-examples 20
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finsage.config import get_settings
from finsage.evaluation import visualization
from finsage.evaluation.comparison import ModelComparison
from finsage.evaluation.generators import (
    AdapterGenerator,
    BaseGenerator,
    MergedModelGenerator,
    MockGenerator,
)
from finsage.evaluation.report_generator import BenchmarkReportGenerator
from finsage.evaluation.runner import EvalRunner
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Evaluate the fine-tuned model and benchmark it.", add_completion=False)
logger = get_logger(__name__)
console = Console()


def _build_generator(
    backend: str,
    model_id: str,
    adapter_path: str | None,
    merged_model_path: str | None,
    device: str,
    load_in_4bit: bool,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> BaseGenerator:
    """Construct the requested fine-tuned generator backend.

    Args:
        backend: ``mock``, ``adapter``, or ``merged``.
        model_id: Base model id (adapter backend).
        adapter_path: LoRA adapter directory (adapter backend).
        merged_model_path: Merged model directory (merged backend).
        device: Device placement.
        load_in_4bit: Whether to load the base model in 4-bit (adapter backend).
        batch_size: Generation batch size.
        max_new_tokens: Max new tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.

    Returns:
        A generator instance.

    Raises:
        ValueError: If the backend is unknown or required paths are missing.
        FileNotFoundError: If the adapter/merged path does not exist.
    """
    if backend == "mock":
        return MockGenerator(quality="finetuned")
    if backend == "adapter":
        if not adapter_path:
            raise ValueError("--adapter-path is required for the adapter backend.")
        return AdapterGenerator(
            model_id=model_id,
            adapter_path=adapter_path,
            device=device,
            load_in_4bit=load_in_4bit,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    if backend == "merged":
        if not merged_model_path:
            raise ValueError("--merged-model-path is required for the merged backend.")
        return MergedModelGenerator(
            merged_model_path=merged_model_path,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    raise ValueError(f"Unknown backend {backend!r}; expected mock|adapter|merged.")


@app.command()
def main(
    test_file: str = typer.Option(..., "--test-file", help="JSONL test set path."),
    baseline_results: str = typer.Option(..., "--baseline-results", help="Baseline results JSON."),
    baseline_predictions: str = typer.Option(
        ..., "--baseline-predictions", help="Baseline predictions JSONL."
    ),
    model_id: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.3", help="Base model id (adapter backend)."
    ),
    adapter_path: str | None = typer.Option(None, help="LoRA adapter directory (adapter backend)."),
    merged_model_path: str | None = typer.Option(
        None, help="Merged model directory (merged backend)."
    ),
    output_dir: str = typer.Option("reports/figures", help="Output directory."),
    backend: str = typer.Option("mock", help="Backend: mock | adapter | merged."),
    max_examples: int | None = typer.Option(None, help="Cap on examples."),
    device: str = typer.Option("auto", help="Device: auto|cuda|cpu."),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit", help="4-bit base."),
    batch_size: int = typer.Option(1, help="Generation batch size."),
    max_new_tokens: int = typer.Option(256, help="Max new tokens."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Nucleus sampling top-p."),
    generate_report: bool = typer.Option(
        True, "--generate-report/--no-generate-report", help="Write the benchmark report."
    ),
    generate_charts: bool = typer.Option(
        True, "--generate-charts/--no-generate-charts", help="Render PNG charts."
    ),
    export_pdf: bool = typer.Option(
        False, "--export-pdf/--no-export-pdf", help="Export the report to PDF (needs pandoc)."
    ),
    report_path: str = typer.Option("reports/benchmark_report.md", help="Report output path."),
    faithfulness: str = typer.Option("lexical", help="Faithfulness metric: lexical | nli."),
) -> None:
    """Evaluate the fine-tuned model, compare to baseline, and write outputs.

    Args:
        test_file: JSONL test set path (must match the baseline run).
        baseline_results: Path to the baseline results JSON.
        baseline_predictions: Path to the baseline predictions JSONL.
        model_id: Base model id (adapter backend).
        adapter_path: LoRA adapter directory (adapter backend).
        merged_model_path: Merged model directory (merged backend).
        output_dir: Output directory for artifacts.
        backend: ``mock``, ``adapter``, or ``merged``.
        max_examples: Optional cap on examples.
        device: Device placement.
        load_in_4bit: Load base model in 4-bit (adapter backend).
        batch_size: Generation batch size.
        max_new_tokens: Max new tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        generate_report: Whether to write the benchmark report.
        generate_charts: Whether to render charts.
        export_pdf: Whether to export the report to PDF.
        report_path: Markdown report output path.

    Raises:
        typer.Exit: With code 1 on missing inputs or missing dependencies.
    """
    setup_logging(get_settings().log_level)

    for label, path in (
        ("Test file", test_file),
        ("Baseline results", baseline_results),
        ("Baseline predictions", baseline_predictions),
    ):
        if not Path(path).exists():
            console.print(f"[red]{label} not found: {path}[/red]")
            raise typer.Exit(code=1)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        generator = _build_generator(
            backend,
            model_id,
            adapter_path,
            merged_model_path,
            device,
            load_in_4bit,
            batch_size,
            max_new_tokens,
            temperature,
            top_p,
        )
    except (ValueError, FileNotFoundError, ImportError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    runner = EvalRunner(generator=generator, output_dir=output_dir, faithfulness=faithfulness)
    try:
        finetuned_results = runner.run_with_prefix(
            test_file, "finetuned", max_examples=max_examples
        )
    except (ImportError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # surface a clean message
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    comparator = ModelComparison()
    base_results = comparator.load_json(baseline_results)
    base_preds = comparator.load_jsonl(baseline_predictions)
    fine_preds = comparator.load_jsonl(finetuned_results["paths"]["predictions"])

    comparison = comparator.compare_results(base_results, finetuned_results)
    qualitative = comparator.compare_predictions(base_preds, fine_preds, max_examples=10)
    comparator.write_comparison_outputs(comparison, qualitative, output_dir)

    chart_paths: dict[str, str] = {}
    if generate_charts:
        chart_paths = _make_charts(comparison, finetuned_results, Path(output_dir))

    if generate_report:
        report = BenchmarkReportGenerator().generate_benchmark_report(
            base_results,
            finetuned_results,
            comparison,
            qualitative,
            report_path,
            chart_paths or None,
        )
        if export_pdf:
            BenchmarkReportGenerator.optionally_export_pdf(report, Path(report).with_suffix(".pdf"))

    _print_summary(
        comparison, finetuned_results, output_dir, report_path if generate_report else None
    )


def _make_charts(comparison: dict, finetuned_results: dict, output_dir: Path) -> dict[str, str]:
    """Render charts and return a name -> relative-path mapping for the report."""
    charts: dict[str, str] = {}
    overall = visualization.plot_overall_metric_comparison(
        comparison, output_dir / "overall_metric_comparison.png"
    )
    deltas = visualization.plot_task_metric_deltas(
        comparison, output_dir / "task_metric_deltas.png"
    )
    dist = visualization.plot_task_counts(finetuned_results, output_dir / "task_distribution.png")
    for name, path in (
        ("overall_metric_comparison", overall),
        ("task_metric_deltas", deltas),
        ("task_distribution", dist),
    ):
        if path is not None:
            charts[name] = f"figures/{Path(path).name}"
    return charts


def _print_summary(
    comparison: dict, finetuned_results: dict, output_dir: str, report_path: str | None
) -> None:
    """Print a Rich summary of the overall metric deltas."""
    table = Table(title="Base vs FinSage-7B (overall)")
    table.add_column("Metric")
    table.add_column("Base", justify="right")
    table.add_column("Fine-tuned", justify="right")
    table.add_column("Δ", justify="right")
    for metric, d in sorted(comparison.get("overall_comparison", {}).items()):
        table.add_row(
            metric, f"{d['baseline']:.4f}", f"{d['finetuned']:.4f}", f"{d['absolute_delta']:+.4f}"
        )
    console.print(table)
    summary = comparison.get("summary", {})
    console.print(
        f"Improved {summary.get('metrics_improved', 0)}/{summary.get('metrics_compared', 0)} "
        f"metrics · examples={finetuned_results.get('num_examples', 0)}"
    )
    console.print(f"Comparison outputs: {output_dir}")
    if report_path:
        console.print(f"Benchmark report: {report_path}")


if __name__ == "__main__":
    app()
