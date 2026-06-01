"""CLI to validate the instruction dataset (Phase 3).

Validates the train/validation/test JSONL files, checks duplicate ids and
train/test company leakage, verifies task-type coverage, writes a report, and
exits non-zero on failure.

Example::

    python scripts/validate_dataset.py validate \\
        --train-path data/datasets/train.jsonl \\
        --validation-path data/datasets/validation.jsonl \\
        --test-path data/datasets/test.jsonl \\
        --report-path data/datasets/validation_report.json
"""

from __future__ import annotations

import typer
from rich.console import Console

from finsage.config import get_settings
from finsage.data.dataset_validator import DatasetValidator
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Validate the instruction dataset.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.callback()
def _main() -> None:
    """Instruction dataset validator (use the ``validate`` subcommand)."""


@app.command()
def validate(
    train_path: str = typer.Option("data/datasets/train.jsonl", help="Train JSONL path."),
    validation_path: str = typer.Option(
        "data/datasets/validation.jsonl", help="Validation JSONL path."
    ),
    test_path: str = typer.Option("data/datasets/test.jsonl", help="Test JSONL path."),
    report_path: str = typer.Option(
        "data/datasets/validation_report.json", help="Validation report output path."
    ),
) -> None:
    """Validate the dataset splits and write a report.

    Args:
        train_path: Path to ``train.jsonl``.
        validation_path: Path to ``validation.jsonl``.
        test_path: Path to ``test.jsonl``.
        report_path: Destination path for the JSON report.

    Raises:
        typer.Exit: With code 1 if validation fails.
    """
    setup_logging(get_settings().log_level)
    validator = DatasetValidator()
    report = validator.validate_splits(train_path, validation_path, test_path)
    validator.write_validation_report(report, report_path)

    for name, file_report in report["files"].items():
        console.print(
            f"  {name}: {file_report['valid_examples']}/{file_report['total_examples']} valid"
        )
    console.print(f"Duplicate ids across splits: {len(report['duplicate_ids'])}")
    console.print(f"Train/test company overlap: {report['train_test_ticker_overlap']}")
    console.print(f"Task types present: {report['task_types_present']}")
    if report["task_types_missing"]:
        console.print(f"[yellow]Task types missing: {report['task_types_missing']}[/yellow]")

    if report["passed"]:
        console.print("[green]Validation PASSED[/green]")
    else:
        console.print("[red]Validation FAILED[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
