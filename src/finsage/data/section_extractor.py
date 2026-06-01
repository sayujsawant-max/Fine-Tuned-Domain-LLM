"""Extract logical sections (Risk Factors, MD&A, Financials) from filing HTML.

Phase 1 provides a deliberately simple, dependency-light implementation: the
HTML is flattened to text with BeautifulSoup and sliced between ``Item`` headings
with regular expressions. The robust, layout-aware extractor (handling tables,
inline XBRL, and the many filer-specific quirks) arrives in Phase 3.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: Whitespace runs collapse to a single space.
_WHITESPACE_RE = re.compile(r"\s+")


def _html_to_text(html: str) -> str:
    """Flatten HTML to normalised plain text.

    Args:
        html: Raw filing HTML (or plain text).

    Returns:
        Whitespace-normalised text with script/style content removed.
    """
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _slice_between(text: str, start_pattern: str, end_pattern: str) -> str:
    """Return the text between the first match of two heading patterns.

    Args:
        text: The flattened filing text to search.
        start_pattern: Regex marking the start of the section (exclusive).
        end_pattern: Regex marking the end of the section (exclusive). If no end
            match is found, the slice runs to the end of ``text``.

    Returns:
        The extracted (and trimmed) section text, or an empty string if the
        start pattern is not found.
    """
    start = re.search(start_pattern, text, flags=re.IGNORECASE)
    if start is None:
        return ""
    tail = text[start.end() :]
    end = re.search(end_pattern, tail, flags=re.IGNORECASE)
    section = tail[: end.start()] if end else tail
    return section.strip()


class SectionExtractor:
    """Extracts priority sections from a 10-K/10-Q filing.

    The public methods all return a mapping with at least ``section`` and
    ``text`` keys so downstream chunking and dataset-building code has a stable
    shape to rely on.
    """

    def extract_risk_factors(self, html: str) -> dict[str, str]:
        """Extract the Risk Factors section (Item 1A).

        Args:
            html: Raw filing HTML.

        Returns:
            A mapping ``{"section": "Risk Factors", "item": "1A", "text": ...}``.
        """
        text = _html_to_text(html)
        body = _slice_between(text, r"item\s*1a\.?\s*risk\s*factors", r"item\s*1b\.?")
        logger.debug("Extracted %d chars of risk factors", len(body))
        return {"section": "Risk Factors", "item": "1A", "text": body}

    def extract_mda(self, html: str) -> dict[str, str]:
        """Extract Management's Discussion and Analysis (Item 7).

        Args:
            html: Raw filing HTML.

        Returns:
            A mapping ``{"section": "MD&A", "item": "7", "text": ...}``.
        """
        text = _html_to_text(html)
        body = _slice_between(
            text,
            r"item\s*7\.?\s*management.?s\s*discussion",
            r"item\s*7a\.?",
        )
        logger.debug("Extracted %d chars of MD&A", len(body))
        return {"section": "MD&A", "item": "7", "text": body}

    def extract_financial_statements(self, html: str) -> dict[str, str]:
        """Extract Financial Statements and Notes (Item 8).

        Args:
            html: Raw filing HTML.

        Returns:
            A mapping ``{"section": "Financial Statements", "item": "8", "text": ...}``.
        """
        text = _html_to_text(html)
        body = _slice_between(
            text,
            r"item\s*8\.?\s*financial\s*statements",
            r"item\s*9\.?",
        )
        logger.debug("Extracted %d chars of financial statements", len(body))
        return {"section": "Financial Statements", "item": "8", "text": body}
