"""CLI to validate the instruction dataset (Phase 4).

Checks each JSONL record carries the required fields and a known task type, and
reports per-split counts. Implemented enough to be useful as soon as datasets
exist; gracefully reports when they do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from finsage.config import get_settings
from finsage.data.instruction_builder import TASK_TYPES
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Validate the instruction dataset.", add_completion=False)
logger = get_logger(__name__)

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "instruction",
    "input",
    "output",
    "task_type",
    "metadata",
)


def validate_record(record: dict[str, object]) -> list[str]:
    """Validate a single dataset record.

    Args:
        record: A parsed JSONL record.

    Returns:
        A list of human-readable error strings; empty if the record is valid.
    """
    errors: list[str] = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in record:
            errors.append(f"missing field '{field_name}'")
    task_type = record.get("task_type")
    if task_type is not None and task_type not in TASK_TYPES:
        errors.append(f"unknown task_type '{task_type}'")
    return errors


@app.command()
def run(
    dataset_dir: str = typer.Option("data/datasets", help="Dataset directory."),
) -> None:
    """Validate all ``*.jsonl`` files in ``dataset_dir``.

    Args:
        dataset_dir: Directory containing the JSONL splits.

    Raises:
        typer.Exit: With code 1 if any record fails validation.
    """
    setup_logging(get_settings().log_level)
    directory = Path(dataset_dir)
    files = sorted(directory.glob("*.jsonl")) if directory.exists() else []
    if not files:
        logger.warning("No JSONL files found in %s (build the dataset first).", dataset_dir)
        return

    total_errors = 0
    for path in files:
        count = 0
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                count += 1
                record = json.loads(line)
                for err in validate_record(record):
                    total_errors += 1
                    logger.error("%s:%d %s", path.name, line_no, err)
        logger.info("%s: %d records checked", path.name, count)

    if total_errors:
        logger.error("Validation failed with %d error(s)", total_errors)
        raise typer.Exit(code=1)
    logger.info("All records valid")


if __name__ == "__main__":
    app()
