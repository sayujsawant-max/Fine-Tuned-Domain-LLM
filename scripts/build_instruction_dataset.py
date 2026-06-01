"""CLI to build the JSONL instruction dataset (Phase 3).

Reads the Phase 2 processed manifest, generates template/extractive instruction
examples, splits them without company leakage, and writes the dataset plus
statistics. Deterministic and CPU-only — no model or network access.

Example::

    python scripts/build_instruction_dataset.py build \\
        --processed-manifest-path data/processed/sec/manifest.jsonl \\
        --output-dir data/datasets \\
        --split-strategy company_holdout \\
        --max-examples 10000
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from finsage.config import get_settings
from finsage.data.chunker import FilingChunker
from finsage.data.dataset_builder import DatasetBuilder
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Build the instruction-tuning dataset.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.callback()
def _main() -> None:
    """Instruction dataset builder (use the ``build`` subcommand)."""


@app.command()
def build(
    processed_manifest_path: str = typer.Option(
        "data/processed/sec/manifest.jsonl", help="Phase 2 processed manifest."
    ),
    output_dir: str = typer.Option("data/datasets", help="Dataset output directory."),
    split_strategy: str = typer.Option(
        "company_holdout", help="Split strategy: company_holdout | time_holdout."
    ),
    max_examples: int | None = typer.Option(None, help="Cap on total examples."),
    train_ratio: float = typer.Option(0.85, help="Train fraction."),
    validation_ratio: float = typer.Option(0.10, help="Validation fraction."),
    test_ratio: float = typer.Option(0.05, help="Test fraction."),
    random_seed: int = typer.Option(42, help="Seed for deterministic splitting."),
    max_tokens: int = typer.Option(512, help="Max tokens per chunk."),
    overlap: int = typer.Option(64, help="Chunk overlap in tokens."),
    min_tokens: int = typer.Option(80, help="Minimum tokens per chunk."),
) -> None:
    """Build, split, and write the instruction dataset.

    Args:
        processed_manifest_path: Path to the Phase 2 processed manifest.
        output_dir: Directory for dataset outputs.
        split_strategy: ``company_holdout`` or ``time_holdout``.
        max_examples: Optional cap on total examples.
        train_ratio: Train fraction.
        validation_ratio: Validation fraction.
        test_ratio: Test fraction.
        random_seed: Seed for deterministic splitting.
        max_tokens: Max tokens per chunk.
        overlap: Chunk overlap in tokens.
        min_tokens: Minimum tokens per chunk.

    Raises:
        typer.Exit: With code 1 if the manifest is missing.
    """
    setup_logging(get_settings().log_level)
    chunker = FilingChunker(max_tokens=max_tokens, overlap=overlap, min_tokens=min_tokens)
    builder = DatasetBuilder(
        processed_manifest_path=processed_manifest_path,
        output_dir=output_dir,
        chunker=chunker,
        random_seed=random_seed,
    )

    try:
        summary = builder.build(
            max_examples=max_examples,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            strategy=split_strategy,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red] Run Phase 2 (extract-sections) first.")
        raise typer.Exit(code=1) from exc

    table = Table(title="FinSage Instruction Dataset")
    table.add_column("Split")
    table.add_column("Examples", justify="right")
    for name, size in summary["split_sizes"].items():
        table.add_row(name, str(size))
    console.print(table)

    leakage = summary["leakage_check"]
    status = "[green]PASS[/green]" if leakage["passed"] else "[red]FAIL[/red]"
    console.print(
        f"Leakage (train/test company overlap): {status} {leakage['train_test_ticker_overlap']}"
    )

    missing = sorted(set(summary["expected_task_types"]) - set(summary["task_types"]))
    console.print(f"Task types present: {len(summary['task_types'])}/10")
    if missing:
        console.print(f"[yellow]Task types not present (no source sections): {missing}[/yellow]")
    console.print(f"Outputs in: {output_dir} (stats: {summary['stats_path']})")


if __name__ == "__main__":
    app()
