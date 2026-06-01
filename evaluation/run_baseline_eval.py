"""CLI to run baseline evaluation of the base model (Phase 4).

Two backends:

- ``mock`` (default) — deterministic, no dependencies; for tests/pipeline checks.
- ``transformers`` — real Hugging Face inference; requires ``.[ml,training]``.

Example (mock)::

    python evaluation/run_baseline_eval.py \\
        --test-file tests/fixtures/eval_test_sample.jsonl \\
        --output-dir /tmp/finsage_baseline_eval --backend mock --max-examples 20

Example (real)::

    python evaluation/run_baseline_eval.py \\
        --test-file data/datasets/test.jsonl \\
        --model-id mistralai/Mistral-7B-Instruct-v0.3 \\
        --output-dir reports/figures --backend transformers \\
        --device auto --load-in-4bit --max-examples 200
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finsage.config import get_settings
from finsage.evaluation.generators import BaseGenerator, MockGenerator, TransformersGenerator
from finsage.evaluation.report_generator import BaselineReportGenerator
from finsage.evaluation.runner import EvalRunner
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Run baseline evaluation of the base model.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.command()
def main(
    test_file: str = typer.Option(..., "--test-file", help="JSONL test set path."),
    model_id: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.3", help="Base model id (transformers backend)."
    ),
    output_dir: str = typer.Option("reports/figures", help="Output directory."),
    backend: str = typer.Option("mock", help="Backend: mock | transformers."),
    max_examples: int | None = typer.Option(None, help="Cap on examples."),
    device: str = typer.Option("auto", help="Device (transformers): auto|cuda|cpu."),
    load_in_4bit: bool = typer.Option(False, help="Load model in 4-bit (transformers)."),
    batch_size: int = typer.Option(1, help="Generation batch size."),
    max_new_tokens: int = typer.Option(256, help="Max new tokens (transformers)."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Nucleus sampling top-p."),
    save_every: int = typer.Option(25, help="Checkpoint predictions every N examples."),
    faithfulness: str = typer.Option("lexical", help="Faithfulness metric: lexical | nli."),
    report_path: str = typer.Option(
        "reports/baseline_eval_report.md", help="Markdown report output path."
    ),
) -> None:
    """Run baseline evaluation and write predictions, metrics, and a report.

    Args:
        test_file: Path to the JSONL test set.
        model_id: Base model id (used by the transformers backend).
        output_dir: Directory for output files.
        backend: ``mock`` or ``transformers``.
        max_examples: Optional cap on the number of examples.
        device: Device placement for the transformers backend.
        load_in_4bit: Load the model in 4-bit (transformers backend).
        batch_size: Generation batch size.
        max_new_tokens: Maximum new tokens per generation (transformers).
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        save_every: Checkpoint cadence in examples.
        report_path: Destination for the Markdown report.

    Raises:
        typer.Exit: With code 1 on a fatal error (missing file, missing deps).
    """
    setup_logging(get_settings().log_level)

    if not Path(test_file).exists():
        console.print(f"[red]Test file not found: {test_file}[/red]")
        raise typer.Exit(code=1)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        generator = _build_generator(
            backend=backend,
            model_id=model_id,
            device=device,
            load_in_4bit=load_in_4bit,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except (ValueError, ImportError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    runner = EvalRunner(
        generator=generator, output_dir=output_dir, save_every=save_every, faithfulness=faithfulness
    )
    try:
        results = runner.run(test_file, max_examples=max_examples)
    except Exception as exc:  # surface a clean message, non-zero exit
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    report = BaselineReportGenerator().generate_markdown_report(
        results, results["paths"]["predictions"], report_path
    )

    _print_summary(results, report)


def _build_generator(
    backend: str,
    model_id: str,
    device: str,
    load_in_4bit: bool,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> BaseGenerator:
    """Construct the requested generator backend.

    Args:
        backend: ``mock`` or ``transformers``.
        model_id: Base model id (transformers).
        device: Device placement.
        load_in_4bit: Whether to load the model in 4-bit.
        batch_size: Generation batch size.
        max_new_tokens: Max new tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.

    Returns:
        A generator instance.

    Raises:
        ValueError: If ``backend`` is unknown.
    """
    if backend == "mock":
        return MockGenerator()
    if backend == "transformers":
        return TransformersGenerator(
            model_id=model_id,
            device=device,
            load_in_4bit=load_in_4bit,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    raise ValueError(f"Unknown backend {backend!r}; expected 'mock' or 'transformers'.")


def _print_summary(results: dict, report_path: Path) -> None:
    """Print a Rich summary table of overall metrics.

    Args:
        results: The aggregated results dict.
        report_path: Path to the written Markdown report.
    """
    table = Table(title=f"Baseline evaluation ({results.get('backend')})")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name, value in sorted(results.get("overall", {}).items()):
        table.add_row(name, f"{float(value):.4f}")
    console.print(table)
    console.print(f"Examples: {results.get('num_examples', 0)}")
    console.print(f"Predictions: {results['paths']['predictions']}")
    console.print(f"Results: {results['paths']['results']}")
    console.print(f"Report: {report_path}")


if __name__ == "__main__":
    app()
