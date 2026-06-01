"""CLI to download SEC EDGAR filings (Phase 2).

Resolves tickers to CIKs, lists 10-K / 10-Q filings, downloads the primary
document of each, and writes a JSONL manifest. Requires ``EDGAR_USER_AGENT``.

Example::

    python scripts/download_edgar.py download \\
        --tickers AAPL MSFT NVDA \\
        --forms 10-K 10-Q \\
        --start-year 2021 --end-year 2023 \\
        --limit-per-company 5 \\
        --output-dir data/raw/sec \\
        --manifest-path data/raw/sec/manifest.jsonl
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from finsage.config import get_settings
from finsage.data.edgar_client import EdgarClient
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Download SEC EDGAR filings.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.callback()
def _main() -> None:
    """SEC EDGAR filing downloader (use the ``download`` subcommand)."""


def _split_tickers(values: list[str]) -> list[str]:
    """Flatten repeated and comma-separated ticker options.

    Args:
        values: Raw ``--tickers`` values (each may be comma-separated).

    Returns:
        A de-duplicated, upper-cased list of ticker symbols.
    """
    out: list[str] = []
    for value in values:
        out.extend(part.strip().upper() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(out))


@app.command()
def download(
    tickers: list[str] = typer.Option(
        ..., "--tickers", "-t", help="Tickers (repeatable or comma-separated)."
    ),
    forms: list[str] = typer.Option(
        ["10-K", "10-Q"], "--forms", "-f", help="Form types to download."
    ),
    start_year: int | None = typer.Option(None, help="Earliest filing year (inclusive)."),
    end_year: int | None = typer.Option(None, help="Latest filing year (inclusive)."),
    limit_per_company: int = typer.Option(5, help="Max filings per company."),
    output_dir: str = typer.Option("data/raw/sec", help="Raw output directory."),
    manifest_path: str = typer.Option(
        "data/raw/sec/manifest.jsonl", help="JSONL manifest output path."
    ),
    force: bool = typer.Option(False, help="Re-download files that already exist."),
) -> None:
    """Download filings for the given tickers and write a manifest.

    Args:
        tickers: Ticker symbols (repeatable or comma-separated).
        forms: Form types to download (e.g. ``10-K``, ``10-Q``).
        start_year: Earliest filing year (inclusive).
        end_year: Latest filing year (inclusive).
        limit_per_company: Maximum filings to download per company.
        output_dir: Root directory for raw filings.
        manifest_path: Destination path for the JSONL manifest.
        force: Re-download files that already exist.

    Raises:
        typer.Exit: With code 1 if ``EDGAR_USER_AGENT`` is not configured.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    ticker_list = _split_tickers(tickers)
    try:
        client = EdgarClient(user_agent=settings.edgar_user_agent)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    rows: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading filings...", total=None)
        for ticker in ticker_list:
            progress.update(task, description=f"Downloading {ticker}...")
            rows.extend(
                client.download_filings(
                    tickers=[ticker],
                    forms=forms,
                    start_year=start_year,
                    end_year=end_year,
                    limit_per_company=limit_per_company,
                    output_dir=output_dir,
                    force=force,
                )
            )
    client.close()

    EdgarClient.write_manifest(rows, manifest_path)
    downloaded = sum(1 for r in rows if r.get("downloaded"))
    console.print(
        f"[green]Done.[/green] {downloaded}/{len(rows)} filing(s) downloaded across "
        f"{len(ticker_list)} ticker(s). Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    app()
