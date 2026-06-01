"""CLI to extract priority sections from raw filings (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import typer

from finsage.config import get_settings
from finsage.data.section_extractor import SectionExtractor
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Extract sections from raw filings.", add_completion=False)
logger = get_logger(__name__)


@app.command()
def run(
    raw_dir: str = typer.Option("data/raw", help="Directory of raw filing HTML."),
    out_dir: str = typer.Option("data/processed", help="Output directory."),
) -> None:
    """Extract Risk Factors, MD&A, and Financial Statements from raw filings.

    Args:
        raw_dir: Directory containing raw filing HTML files.
        out_dir: Directory to write extracted section JSON into.
    """
    setup_logging(get_settings().log_level)
    extractor = SectionExtractor()
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("*.htm*")) if raw_path.exists() else []
    logger.info(
        "Phase 3 stub: found %d raw filing(s) in %s; output -> %s " "(extractor=%s)",
        len(files),
        raw_dir,
        out_dir,
        type(extractor).__name__,
    )


if __name__ == "__main__":
    app()
