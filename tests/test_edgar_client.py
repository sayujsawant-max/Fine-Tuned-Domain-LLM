"""Tests for EdgarClient (all network access mocked via conftest)."""

from __future__ import annotations

from pathlib import Path

import pytest

from finsage.data.edgar_client import EdgarClient

CIK_AAPL = "0000320193"


def test_normalizes_cik():
    """CIK normalisation zero-pads to 10 digits from int or string input."""
    assert EdgarClient.normalize_cik(320193) == CIK_AAPL
    assert EdgarClient.normalize_cik("320193") == CIK_AAPL
    assert EdgarClient.normalize_cik("0000320193") == CIK_AAPL
    assert EdgarClient.normalize_cik("CIK0000320193") == CIK_AAPL


def test_missing_user_agent_raises_helpful_error(monkeypatch):
    """Constructing without a user agent raises a helpful ValueError."""
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="EDGAR_USER_AGENT"):
        EdgarClient(user_agent=None)


def test_resolves_ticker_to_cik(edgar_client):
    """Tickers resolve to CIKs case-insensitively from the mocked mapping."""
    assert edgar_client.cik_for_ticker("AAPL") == CIK_AAPL
    assert edgar_client.cik_for_ticker("aapl") == CIK_AAPL
    assert edgar_client.cik_for_ticker("MSFT") == "0000789019"


def test_unknown_ticker_raises(edgar_client):
    """An unknown ticker raises KeyError."""
    with pytest.raises(KeyError):
        edgar_client.cik_for_ticker("NOPE")


def test_builds_correct_document_url(edgar_client):
    """document_url is built from CIK, accession, and primary document."""
    filings = edgar_client.list_filings(CIK_AAPL, forms=["10-K"])
    fy2022 = next(f for f in filings if f["accession_number"] == "0000320193-22-000108")
    assert (
        fy2022["document_url"]
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm"
    )
    assert fy2022["accession_number_no_dashes"] == "000032019322000108"


def test_filters_by_form(edgar_client):
    """Form filtering keeps only the requested form types."""
    filings = edgar_client.list_filings(CIK_AAPL, forms=["10-K"])
    assert len(filings) == 3
    assert {f["form"] for f in filings} == {"10-K"}


def test_filters_by_year(edgar_client):
    """Year filtering uses the filing date and is inclusive."""
    filings = edgar_client.list_filings(CIK_AAPL, forms=["10-K"], start_year=2022, end_year=2023)
    years = {f["filing_date"][:4] for f in filings}
    assert years == {"2022", "2023"}
    assert "2021" not in years


def test_respects_limit_and_sorts_newest_first(edgar_client):
    """Listing respects the limit and returns newest filings first."""
    filings = edgar_client.list_filings(CIK_AAPL, forms=["10-K", "10-Q"], limit=2)
    assert len(filings) == 2
    assert filings[0]["filing_date"] >= filings[1]["filing_date"]


def test_list_filings_uses_mock_not_network(edgar_client):
    """The mocked transport supplies data — no real SEC call is made."""
    tickers = edgar_client.get_company_tickers()
    assert len(tickers) == 3  # only the three fixture companies


def test_download_filing_writes_file(edgar_client, tmp_path: Path):
    """download_filing writes the document to the expected path layout."""
    filings = edgar_client.list_filings(CIK_AAPL, forms=["10-K"], limit=1)
    filings[0]["ticker"] = "AAPL"
    out_dir = tmp_path / "raw"

    path = edgar_client.download_filing(filings[0], output_dir=out_dir)

    assert path.exists()
    assert path.suffix == ".html"
    assert "AAPL" in path.parts and "10-K" in path.parts
    assert "ACME WIDGETS" in path.read_text(encoding="utf-8")


def test_download_filing_skips_existing(edgar_client, tmp_path: Path):
    """An existing file is not re-downloaded unless force=True."""
    filings = edgar_client.list_filings(CIK_AAPL, forms=["10-K"], limit=1)
    out_dir = tmp_path / "raw"
    path = edgar_client.download_filing(filings[0], output_dir=out_dir)
    path.write_text("CACHED", encoding="utf-8")

    again = edgar_client.download_filing(filings[0], output_dir=out_dir)
    assert again.read_text(encoding="utf-8") == "CACHED"  # untouched


def test_write_manifest_is_valid_jsonl(edgar_client, tmp_path: Path):
    """The download manifest is written as valid JSONL."""
    import json

    rows = edgar_client.download_filings(
        tickers=["AAPL"], forms=["10-K"], limit_per_company=2, output_dir=tmp_path / "raw"
    )
    manifest = EdgarClient.write_manifest(rows, tmp_path / "manifest.jsonl")
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert all("document_url" in row and "raw_path" in row for row in lines)
