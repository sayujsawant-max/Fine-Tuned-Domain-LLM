"""Shared pytest fixtures.

Provides an :class:`EdgarClient` wired to an ``httpx.MockTransport`` so tests
exercise the real client logic without ever touching the SEC network.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from finsage.data.edgar_client import EdgarClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    """Read a fixture file's text.

    Args:
        name: File name within the fixtures directory.

    Returns:
        The file contents as a string.
    """
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def sample_10k_html() -> str:
    """Return the sample 10-K HTML fixture."""
    return _load_fixture("sample_10k.html")


@pytest.fixture
def processed_manifest_path() -> Path:
    """Return the path to the sample processed manifest fixture."""
    return FIXTURES_DIR / "processed_manifest_sample.jsonl"


@pytest.fixture
def mock_transport() -> httpx.MockTransport:
    """Return a MockTransport routing SEC URLs to local fixtures."""
    tickers = _load_fixture("company_tickers_sample.json")
    submissions = _load_fixture("submissions_sample.json")
    sample_html = _load_fixture("sample_10k.html")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, text=tickers)
        if "/submissions/CIK" in url:
            return httpx.Response(200, text=submissions)
        if "/Archives/edgar/data/" in url:
            return httpx.Response(200, text=sample_html)
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


@pytest.fixture
def edgar_client(tmp_path: Path, mock_transport: httpx.MockTransport) -> EdgarClient:
    """Return an EdgarClient backed by the mock transport (no network)."""
    client = EdgarClient(
        user_agent="FinSage Test test@example.com",
        rate_limit_per_second=0,  # disable throttling in tests
        cache_dir=tmp_path / "cache",
    )
    client._client = httpx.Client(
        transport=mock_transport,
        headers={"User-Agent": client.user_agent},
    )
    return client
