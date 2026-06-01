"""CLI to extract filing sections into clean per-section text files (Phase 2).

Reads the raw manifest produced by ``download_edgar.py``, extracts the five
target sections from each filing, writes one ``.txt`` per section, and writes a
processed manifest.

Example::

    python scripts/extract_sections.py extract \\
        --manifest-path data/raw/sec/manifest.jsonl \\
        --output-dir data/processed/sec \\
        --processed-manifest-path data/processed/sec/manifest.jsonl
"""

from __future__ import annotations

import typer
from rich.console import Console

from finsage.config import get_settings
from finsage.data.preprocessor import FilingPreprocessor
from finsage.data.section_extractor import TARGET_SECTIONS
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Extract sections from raw filings.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.callback()
def _main() -> None:
    """Filing section extractor (use the ``extract`` subcommand)."""


@app.command()
def extract(
    manifest_path: str = typer.Option(
        "data/raw/sec/manifest.jsonl", help="Raw manifest (from the downloader)."
    ),
    output_dir: str = typer.Option("data/processed/sec", help="Processed output directory."),
    processed_manifest_path: str = typer.Option(
        "data/processed/sec/manifest.jsonl", help="Processed manifest output path."
    ),
) -> None:
    """Process all raw filings in the manifest into per-section text files.

    Args:
        manifest_path: Path to the raw JSONL manifest.
        output_dir: Root directory for processed output.
        processed_manifest_path: Destination path for the processed manifest.

    Raises:
        typer.Exit: With code 1 if the raw manifest does not exist.
    """
    setup_logging(get_settings().log_level)
    preprocessor = FilingPreprocessor()

    try:
        rows = preprocessor.process_manifest(manifest_path, output_dir=output_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red] Run the downloader first.")
        raise typer.Exit(code=1) from exc

    preprocessor.write_manifest(rows, processed_manifest_path)

    filings = len({r["raw_path"] for r in rows})
    total_words = sum(int(r["text_words"]) for r in rows)
    found_sections = {r["section"] for r in rows}
    missing = len(TARGET_SECTIONS) - len(found_sections) if rows else len(TARGET_SECTIONS)

    console.print("[green]Extraction complete.[/green]")
    console.print(f"  filings processed:   {filings}")
    console.print(f"  sections extracted:  {len(rows)}")
    console.print(f"  distinct sections:   {sorted(found_sections)}")
    console.print(f"  missing section types: {missing}")
    console.print(f"  total words:         {total_words}")
    console.print(f"  processed manifest:  {processed_manifest_path}")


if __name__ == "__main__":
    app()
