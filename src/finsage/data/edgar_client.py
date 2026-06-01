"""Client for fetching filings from SEC EDGAR.

Implements the SEC fair-access rules: a mandatory descriptive ``User-Agent``,
conservative client-side rate limiting, retries with backoff on transient
errors, and on-disk caching of the (large, rarely-changing) JSON metadata so we
never re-download it. All network access is isolated in this module.

SEC endpoints used:

- Company tickers:  ``https://www.sec.gov/files/company_tickers.json``
- Submissions:      ``https://data.sec.gov/submissions/CIK##########.json``
- Document archive: ``https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}``
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: SEC company ticker -> CIK mapping.
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

#: Per-company submissions (filing history) endpoint template.
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

#: Base URL for the document archive.
ARCHIVE_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

#: HTTP status codes worth retrying.
RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Filing form types this project ingests.
SUPPORTED_FORMS: tuple[str, ...] = ("10-K", "10-Q", "8-K")


class EdgarError(RuntimeError):
    """Raised when an EDGAR request fails after exhausting retries."""


class EdgarClient:
    """Thin, polite client over the SEC EDGAR APIs.

    Args:
        user_agent: Descriptive User-Agent with contact info, as required by SEC
            fair-access rules (e.g. ``"Your Name you@example.com"``). If ``None``,
            the ``EDGAR_USER_AGENT`` environment variable is used.
        rate_limit_per_second: Maximum requests per second (client-side throttle).
            Defaults to a conservative 5, below the SEC's stated ceiling.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries on transient HTTP errors.
        cache_dir: Directory for cached JSON metadata.

    Raises:
        ValueError: If no user agent is provided or found in the environment.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        rate_limit_per_second: float = 5.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        cache_dir: Path | str = "data/cache/edgar",
    ) -> None:
        resolved_agent = user_agent or os.environ.get("EDGAR_USER_AGENT")
        if not resolved_agent or not resolved_agent.strip():
            raise ValueError(
                "A SEC EDGAR User-Agent is required. Set the EDGAR_USER_AGENT "
                "environment variable (or pass user_agent=...) to a descriptive "
                'value with contact info, e.g. EDGAR_USER_AGENT="Your Name '
                'you@example.com". SEC rejects requests without one.'
            )

        self.user_agent = resolved_agent.strip()
        self.rate_limit_per_second = rate_limit_per_second
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)

        self._min_interval = 1.0 / rate_limit_per_second if rate_limit_per_second > 0 else 0.0
        self._last_request_ts = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> EdgarClient:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager, closing the HTTP client."""
        self.close()

    # ------------------------------------------------------------------
    # Low-level HTTP (the only place that touches the network)
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        """Sleep as needed to respect the configured request rate."""
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _get(self, url: str) -> httpx.Response:
        """Perform a throttled GET with retries on transient errors.

        Args:
            url: The absolute URL to fetch.

        Returns:
            The successful :class:`httpx.Response`.

        Raises:
            EdgarError: If the request fails after exhausting all retries.
        """
        last_error: str = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self._client.get(url)
            except httpx.HTTPError as exc:  # network/timeout errors
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("GET %s failed (attempt %d): %s", url, attempt + 1, last_error)
            else:
                if response.status_code in RETRY_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        "GET %s -> %s (attempt %d), retrying",
                        url,
                        response.status_code,
                        attempt + 1,
                    )
                elif response.is_success:
                    return response
                else:
                    response.raise_for_status()

            if attempt < self.max_retries:
                backoff = min(2.0**attempt * 0.5, 8.0)
                time.sleep(backoff)

        raise EdgarError(f"Failed to GET {url} after {self.max_retries + 1} attempts: {last_error}")

    def _cached_json(self, url: str, cache_name: str, *, force: bool = False) -> Any:
        """Fetch JSON from ``url``, caching the raw body under ``cache_name``.

        Args:
            url: The JSON endpoint to fetch.
            cache_name: Relative path (under ``cache_dir``) for the cached body.
            force: If ``True``, bypass and refresh the cache.

        Returns:
            The parsed JSON payload.
        """
        cache_path = self.cache_dir / cache_name
        if cache_path.exists() and not force:
            logger.debug("Cache hit: %s", cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        response = self._get(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(response.text, encoding="utf-8")
        logger.debug("Cached %s -> %s", url, cache_path)
        return response.json()

    # ------------------------------------------------------------------
    # CIK / ticker resolution
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_cik(cik: str | int) -> str:
        """Return a 10-digit, zero-padded CIK string.

        Args:
            cik: A CIK as an int or string (with or without leading zeros).

        Returns:
            The CIK as a 10-character zero-padded string.

        Raises:
            ValueError: If ``cik`` contains no digits.
        """
        digits = "".join(ch for ch in str(cik) if ch.isdigit())
        if not digits:
            raise ValueError(f"Invalid CIK: {cik!r}")
        return digits.zfill(10)

    def get_company_tickers(self, force: bool = False) -> dict[str, Any]:
        """Fetch and cache the SEC company-tickers mapping.

        Args:
            force: If ``True``, refresh the cache.

        Returns:
            The parsed ``company_tickers.json`` payload (index -> record).
        """
        data = self._cached_json(COMPANY_TICKERS_URL, "company_tickers.json", force=force)
        return dict(data)

    def cik_for_ticker(self, ticker: str) -> str:
        """Resolve a ticker symbol to its 10-digit CIK.

        Args:
            ticker: The ticker symbol (case-insensitive).

        Returns:
            The 10-digit zero-padded CIK.

        Raises:
            KeyError: If the ticker is not found in the SEC mapping.
        """
        target = ticker.strip().upper()
        for record in self.get_company_tickers().values():
            if str(record.get("ticker", "")).upper() == target:
                return self.normalize_cik(record["cik_str"])
        raise KeyError(f"Ticker {ticker!r} not found in SEC company_tickers.json")

    # ------------------------------------------------------------------
    # Filing listing
    # ------------------------------------------------------------------
    def get_filing_index(self, cik: str | int, force: bool = False) -> dict[str, Any]:
        """Fetch and cache the submissions (filing history) for a company.

        Args:
            cik: The company CIK (int or string).
            force: If ``True``, refresh the cache.

        Returns:
            The parsed submissions JSON payload.
        """
        normalized = self.normalize_cik(cik)
        url = SUBMISSIONS_URL.format(cik=normalized)
        return dict(self._cached_json(url, f"submissions/CIK{normalized}.json", force=force))

    def list_filings(
        self,
        cik: str | int,
        forms: list[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List filings for a company, filtered by form and year.

        Args:
            cik: The company CIK (int or string).
            forms: Form types to keep (e.g. ``["10-K", "10-Q"]``). ``None`` keeps all.
            start_year: Earliest filing year (inclusive), by filing date.
            end_year: Latest filing year (inclusive), by filing date.
            limit: Maximum number of filings to return (after sorting newest first).

        Returns:
            A list of filing metadata dicts, newest first. Each dict contains
            ``cik``, ``accession_number``, ``accession_number_no_dashes``,
            ``filing_date``, ``report_date``, ``form``, ``primary_document``,
            ``filing_url``, ``document_url``, and ``ticker`` (if known).
        """
        normalized = self.normalize_cik(cik)
        data = self.get_filing_index(normalized)
        tickers = data.get("tickers") or []
        ticker = str(tickers[0]).upper() if tickers else None

        recent = data.get("filings", {}).get("recent", {})
        accession_numbers = recent.get("accessionNumber", [])
        forms_arr = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        wanted_forms = {f.upper() for f in forms} if forms else None
        rows: list[dict[str, Any]] = []
        for i, accession in enumerate(accession_numbers):
            form = forms_arr[i] if i < len(forms_arr) else ""
            if wanted_forms is not None and form.upper() not in wanted_forms:
                continue

            filing_date = filing_dates[i] if i < len(filing_dates) else ""
            year = int(filing_date[:4]) if filing_date[:4].isdigit() else None
            if start_year is not None and (year is None or year < start_year):
                continue
            if end_year is not None and (year is None or year > end_year):
                continue

            primary_document = primary_docs[i] if i < len(primary_docs) else ""
            accession_no_dashes = accession.replace("-", "")
            cik_no_zeros = str(int(normalized))
            folder = f"{ARCHIVE_BASE_URL}/{cik_no_zeros}/{accession_no_dashes}"
            rows.append(
                {
                    "cik": normalized,
                    "ticker": ticker,
                    "accession_number": accession,
                    "accession_number_no_dashes": accession_no_dashes,
                    "filing_date": filing_date,
                    "report_date": report_dates[i] if i < len(report_dates) else "",
                    "form": form,
                    "primary_document": primary_document,
                    "filing_url": f"{folder}/{accession}-index.htm",
                    "document_url": f"{folder}/{primary_document}",
                }
            )

        rows.sort(key=lambda r: r["filing_date"], reverse=True)
        if limit is not None:
            rows = rows[:limit]
        logger.info("Listed %d filing(s) for CIK %s", len(rows), normalized)
        return rows

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------
    @staticmethod
    def _filing_path(filing: dict[str, Any], output_dir: Path) -> Path:
        """Compute the on-disk path for a filing's primary document.

        Args:
            filing: A filing metadata dict from :meth:`list_filings`.
            output_dir: The root output directory.

        Returns:
            The destination path:
            ``{output_dir}/{ticker_or_cik}/{form}/{year}/{accession}.html``.
        """
        ticker_or_cik = filing.get("ticker") or filing["cik"]
        form = str(filing.get("form", "UNKNOWN")).replace("/", "-")
        year = str(filing.get("filing_date", "0000"))[:4] or "0000"
        accession = filing["accession_number_no_dashes"]
        return Path(output_dir) / str(ticker_or_cik) / form / year / f"{accession}.html"

    def download_filing(
        self,
        filing: dict[str, Any],
        output_dir: Path | str = "data/raw/sec",
        force: bool = False,
    ) -> Path:
        """Download a filing's primary document to disk.

        Args:
            filing: A filing metadata dict from :meth:`list_filings`.
            output_dir: Root directory for raw filings.
            force: If ``True``, re-download even if the file already exists.

        Returns:
            The path to the saved HTML file.
        """
        dest = self._filing_path(filing, Path(output_dir))
        if dest.exists() and not force:
            logger.debug("Skipping existing %s", dest)
            return dest

        response = self._get(filing["document_url"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(response.text, encoding="utf-8")
        logger.info("Downloaded %s", dest)
        return dest

    def download_filings(
        self,
        tickers: list[str],
        forms: list[str],
        start_year: int | None = None,
        end_year: int | None = None,
        limit_per_company: int = 5,
        output_dir: Path | str = "data/raw/sec",
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Download filings for several companies and return manifest rows.

        Args:
            tickers: Ticker symbols to download for.
            forms: Form types to include (e.g. ``["10-K", "10-Q"]``).
            start_year: Earliest filing year (inclusive).
            end_year: Latest filing year (inclusive).
            limit_per_company: Maximum filings to download per company.
            output_dir: Root directory for raw filings.
            force: If ``True``, re-download existing files.

        Returns:
            A list of manifest rows (filing metadata augmented with ``raw_path``
            and ``downloaded``). Companies that fail to resolve are skipped with a
            logged warning rather than aborting the whole run.
        """
        manifest: list[dict[str, Any]] = []
        for ticker in tickers:
            try:
                cik = self.cik_for_ticker(ticker)
            except KeyError:
                logger.warning("Skipping unknown ticker %s", ticker)
                continue

            filings = self.list_filings(
                cik,
                forms=forms,
                start_year=start_year,
                end_year=end_year,
                limit=limit_per_company,
            )
            for filing in filings:
                filing["ticker"] = ticker.upper()
                try:
                    path = self.download_filing(filing, output_dir=output_dir, force=force)
                    row = {**filing, "raw_path": str(path), "downloaded": True}
                except (EdgarError, httpx.HTTPError) as exc:
                    logger.warning("Failed to download %s: %s", filing["document_url"], exc)
                    row = {**filing, "raw_path": "", "downloaded": False}
                manifest.append(row)

        logger.info("Downloaded %d filing(s) across %d ticker(s)", len(manifest), len(tickers))
        return manifest

    @staticmethod
    def write_manifest(rows: list[dict[str, Any]], path: Path | str) -> Path:
        """Write manifest rows as JSONL.

        Args:
            rows: The manifest rows to write.
            path: Destination JSONL path.

        Returns:
            The path written to.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote manifest with %d row(s) to %s", len(rows), out_path)
        return out_path
