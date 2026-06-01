"""CLI to download SEC EDGAR filings (Phase 2).

Phase 1 wires the Typer command and logging; the actual download is delegated to
:class:`finsage.data.edgar_client.EdgarClient`, which raises until Phase 2.
"""

from __future__ import annotations

import typer

from finsage.config import get_settings
from finsage.data.edgar_client import EdgarClient
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Download SEC EDGAR filings.", add_completion=False)
logger = get_logger(__name__)


@app.command()
def run(
    cik: str = typer.Option(..., help="Central Index Key of the filer."),
    form_type: str = typer.Option("10-K", help="Filing form type."),
    limit: int = typer.Option(5, help="Maximum number of filings to download."),
    dest: str = typer.Option("data/raw", help="Destination directory."),
) -> None:
    """Download recent filings for a company into ``dest``.

    Args:
        cik: Central Index Key of the filer.
        form_type: Filing form type to download.
        limit: Maximum number of filings.
        dest: Destination directory for raw filings.
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Phase 2 stub: would download %d %s filings for CIK %s into %s", limit, form_type, cik, dest
    )
    client = EdgarClient(user_agent=settings.edgar_user_agent)
    index = client.get_filing_index(cik=cik, form_type=form_type, limit=limit)
    for filing in index:
        client.download_filing(filing, dest)


if __name__ == "__main__":
    app()
