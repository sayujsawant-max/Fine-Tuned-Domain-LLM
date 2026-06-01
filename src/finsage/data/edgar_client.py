"""Client for fetching filings from SEC EDGAR.

Phase 1 ships the public interface only. The network-facing methods raise
:class:`NotImplementedError` so the contract is explicit and tested, while the
real implementation (rate limiting, the mandatory descriptive ``User-Agent``,
the submissions JSON API) lands in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: Base URL for the SEC EDGAR data API.
EDGAR_BASE_URL = "https://data.sec.gov"

#: Filing form types this project ingests.
SUPPORTED_FORMS: tuple[str, ...] = ("10-K", "10-Q", "8-K")


@dataclass
class FilingRef:
    """A reference to a single filing on EDGAR.

    Attributes:
        cik: Central Index Key of the filer (zero-padded to 10 digits).
        accession_number: EDGAR accession number (e.g. ``0000320193-22-000108``).
        form_type: Filing form type, one of :data:`SUPPORTED_FORMS`.
        filing_date: Filing date in ISO ``YYYY-MM-DD`` format.
        primary_document: File name of the primary filing document.
    """

    cik: str
    accession_number: str
    form_type: str
    filing_date: str
    primary_document: str


@dataclass
class EdgarClient:
    """Thin client over the SEC EDGAR data API.

    Args:
        user_agent: Descriptive User-Agent string with contact info, as required
            by SEC fair-access rules (e.g. ``"FinSage Research you@example.com"``).
        base_url: API base URL. Override only for testing.
        request_delay_seconds: Minimum delay between requests to stay within the
            SEC's ~10 requests/second fair-access limit.
    """

    user_agent: str | None = None
    base_url: str = EDGAR_BASE_URL
    request_delay_seconds: float = 0.2
    _headers: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Build default request headers from the configured user agent."""
        if self.user_agent:
            self._headers = {"User-Agent": self.user_agent}

    def list_companies(self, query: str | None = None) -> list[dict[str, str]]:
        """List companies known to EDGAR, optionally filtered by a query.

        Args:
            query: Optional case-insensitive substring to match against the
                company name or ticker.

        Returns:
            A list of ``{"cik", "ticker", "title"}`` mappings.

        Raises:
            NotImplementedError: Always, until Phase 2.
        """
        raise NotImplementedError("EdgarClient.list_companies lands in Phase 2.")

    def get_filing_index(
        self, cik: str, form_type: str = "10-K", limit: int = 10
    ) -> list[FilingRef]:
        """Return recent filings for a company.

        Args:
            cik: Central Index Key of the filer.
            form_type: Filing form type to filter on. Defaults to ``"10-K"``.
            limit: Maximum number of filings to return.

        Returns:
            A list of :class:`FilingRef` ordered most-recent first.

        Raises:
            NotImplementedError: Always, until Phase 2.
        """
        raise NotImplementedError("EdgarClient.get_filing_index lands in Phase 2.")

    def download_filing(self, filing: FilingRef, dest_dir: str | Path) -> Path:
        """Download the primary document of a filing to disk.

        Args:
            filing: The filing reference to download.
            dest_dir: Directory to write the raw document into. Created if absent.

        Returns:
            The path to the downloaded file.

        Raises:
            NotImplementedError: Always, until Phase 2.
        """
        raise NotImplementedError("EdgarClient.download_filing lands in Phase 2.")
