"""CLI to build the JSONL instruction dataset (Phase 4)."""

from __future__ import annotations

import typer

from finsage.config import get_settings
from finsage.data.instruction_builder import TASK_TYPES, InstructionBuilder
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Build the instruction-tuning dataset.", add_completion=False)
logger = get_logger(__name__)


@app.command()
def run(
    processed_dir: str = typer.Option("data/processed", help="Directory of extracted sections."),
    out_dir: str = typer.Option("data/datasets", help="Output dataset directory."),
) -> None:
    """Build train/validation/test JSONL splits from processed sections.

    Args:
        processed_dir: Directory containing extracted/chunked sections.
        out_dir: Directory to write ``train.jsonl`` / ``validation.jsonl`` /
            ``test.jsonl`` into.
    """
    setup_logging(get_settings().log_level)
    builder = InstructionBuilder()
    logger.info(
        "Phase 4 stub: would build dataset from %s into %s across %d task types " "(builder=%s)",
        processed_dir,
        out_dir,
        len(TASK_TYPES),
        type(builder).__name__,
    )


if __name__ == "__main__":
    app()
