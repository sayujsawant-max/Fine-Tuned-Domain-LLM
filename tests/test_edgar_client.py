"""Tests for the EdgarClient public interface (Phase 1 placeholders)."""

from __future__ import annotations

import pytest

from finsage.data.edgar_client import EdgarClient, FilingRef


def test_client_initializes_with_user_agent():
    """The client should store the user agent and build a header from it."""
    client = EdgarClient(user_agent="FinSage Research test@example.com")
    assert client.user_agent == "FinSage Research test@example.com"
    assert client._headers["User-Agent"] == "FinSage Research test@example.com"


def test_client_methods_exist_and_raise():
    """Placeholder methods should exist and raise NotImplementedError."""
    client = EdgarClient(user_agent="ua")
    with pytest.raises(NotImplementedError):
        client.list_companies()
    with pytest.raises(NotImplementedError):
        client.get_filing_index(cik="0000320193")
    with pytest.raises(NotImplementedError):
        client.download_filing(
            FilingRef("0000320193", "acc", "10-K", "2022-10-28", "doc.htm"),
            "data/raw",
        )
