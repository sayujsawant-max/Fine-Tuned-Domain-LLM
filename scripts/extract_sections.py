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

from pathlib import Path

import typer
import yaml
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
    config: str = typer.Option("configs/data_config.yaml", help="Data config YAML."),
    min_section_words: int | None = typer.Option(
        None, help="Drop sections below this word count; default from config."
    ),
) -> None:
    """Process all raw filings in the manifest into per-section text files.

    Args:
        manifest_path: Path to the raw JSONL manifest.
        output_dir: Root directory for processed output.
        processed_manifest_path: Destination path for the processed manifest.
        config: Path to the data config YAML (for ``sections.min_section_words``).
        min_section_words: Override for the minimum section word count.

    Raises:
        typer.Exit: With code 1 if the raw manifest does not exist.
    """
    setup_logging(get_settings().log_level)

    threshold = min_section_words
    if threshold is None:
        config_path = Path(config)
        if config_path.exists():
            data_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            threshold = int(data_cfg.get("sections", {}).get("min_section_words", 0))
        else:
            threshold = 0

    preprocessor = FilingPreprocessor(min_section_words=threshold)
    logger.info("Using min_section_words=%d", threshold)

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
