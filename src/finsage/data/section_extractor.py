"""Extract the major sections of a 10-K / 10-Q filing from raw HTML.

SEC filings vary enormously in markup, so extraction works on cleaned plain
text and uses robust regex detection of ``Item`` headings. For each target item
we take the span from its heading to the next *different* item heading, and —
to skip short table-of-contents entries — keep the candidate span that yields
the most text. Missing sections are skipped, never fatal.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: Matches an "Item N" / "Item NA" heading; captures the number and optional letter.
_ITEM_ANCHOR_RE = re.compile(r"\bitem\s+(\d{1,2})\s*([a-z])?\b", re.IGNORECASE)

#: Collapses runs of inline whitespace (spaces, tabs, non-breaking spaces) but
#: preserves newlines so ``Item`` headings stay on their own lines.
_INLINE_WS_RE = re.compile(r"[^\S\n]+")

#: Section name -> (item id, human label, title) for the five target sections.
_SECTION_ITEMS: dict[str, tuple[str, str, str]] = {
    "business": ("1", "1", "Business"),
    "risk_factors": ("1a", "1A", "Risk Factors"),
    "mda": ("7", "7", "Management's Discussion and Analysis"),
    "market_risk": (
        "7a",
        "7A",
        "Quantitative and Qualitative Disclosures About Market Risk",
    ),
    "financial_statements": ("8", "8", "Financial Statements and Supplementary Data"),
}

#: Public, ordered list of the section names this extractor produces.
TARGET_SECTIONS: tuple[str, ...] = tuple(_SECTION_ITEMS.keys())


def _normalize_item_id(number: str, letter: str | None) -> str:
    """Build a normalized item id such as ``"1a"`` from regex groups.

    Args:
        number: The captured item number (e.g. ``"1"``, ``"7"``).
        letter: The optional captured suffix letter (e.g. ``"a"``), or ``None``.

    Returns:
        The lower-cased item id, e.g. ``"1a"`` or ``"7"``.
    """
    return f"{number}{(letter or '').lower()}"


def _find_anchors(text: str) -> list[tuple[str, int]]:
    """Find all ``Item`` heading anchors in ``text``.

    Args:
        text: Cleaned filing text.

    Returns:
        A list of ``(item_id, start_offset)`` tuples in document order.
    """
    return [
        (_normalize_item_id(m.group(1), m.group(2)), m.start())
        for m in _ITEM_ANCHOR_RE.finditer(text)
    ]


def _extract_item(text: str, item_id: str) -> str:
    """Extract the body of one item from cleaned text.

    For each occurrence of the item heading, the body runs to the next anchor
    with a *different* item id. The longest such body is returned, which skips
    short table-of-contents references in favour of the real section.

    Args:
        text: Cleaned filing text.
        item_id: Normalized item id to extract (e.g. ``"1a"``).

    Returns:
        The extracted section text, or an empty string if not found.
    """
    anchors = _find_anchors(text)
    starts = [pos for (aid, pos) in anchors if aid == item_id]
    if not starts:
        return ""

    best = ""
    for start in starts:
        end = len(text)
        for aid, pos in anchors:
            if pos > start and aid != item_id:
                end = pos
                break
        candidate = text[start:end].strip()
        if len(candidate) > len(best):
            best = candidate
    return best


class SectionExtractor:
    """Extracts the five priority sections from a 10-K / 10-Q filing."""

    def clean_html(self, html: str) -> str:
        """Convert filing HTML to clean, heading-preserving plain text.

        Removes ``script`` / ``style`` tags, extracts readable text, drops
        non-breaking spaces, collapses inline whitespace, and keeps one line per
        text block so ``Item`` headings remain detectable.

        Args:
            html: Raw filing HTML (or plain text).

        Returns:
            Cleaned plain text.
        """
        soup = BeautifulSoup(html or "", "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()

        raw = soup.get_text(separator="\n")
        lines = [_INLINE_WS_RE.sub(" ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)

    def _extract(self, html: str, name: str) -> dict[str, str]:
        """Extract a single named section into a metadata dict.

        Args:
            html: Raw filing HTML.
            name: One of :data:`TARGET_SECTIONS`.

        Returns:
            A mapping with ``section``, ``item``, ``title``, and ``text`` keys.
        """
        item_id, label, title = _SECTION_ITEMS[name]
        body = _extract_item(self.clean_html(html), item_id)
        logger.debug("Extracted %d chars for section %s (item %s)", len(body), name, label)
        return {"section": name, "item": label, "title": title, "text": body}

    def extract_business(self, html: str) -> dict[str, str]:
        """Extract Item 1 — Business.

        Args:
            html: Raw filing HTML.

        Returns:
            A section dict (see :meth:`_extract`).
        """
        return self._extract(html, "business")

    def extract_risk_factors(self, html: str) -> dict[str, str]:
        """Extract Item 1A — Risk Factors.

        Args:
            html: Raw filing HTML.

        Returns:
            A section dict (see :meth:`_extract`).
        """
        return self._extract(html, "risk_factors")

    def extract_mda(self, html: str) -> dict[str, str]:
        """Extract Item 7 — Management's Discussion and Analysis.

        Args:
            html: Raw filing HTML.

        Returns:
            A section dict (see :meth:`_extract`).
        """
        return self._extract(html, "mda")

    def extract_market_risk(self, html: str) -> dict[str, str]:
        """Extract Item 7A — Quantitative and Qualitative Disclosures About Market Risk.

        Args:
            html: Raw filing HTML.

        Returns:
            A section dict (see :meth:`_extract`).
        """
        return self._extract(html, "market_risk")

    def extract_financial_statements(self, html: str) -> dict[str, str]:
        """Extract Item 8 — Financial Statements and Supplementary Data.

        Args:
            html: Raw filing HTML.

        Returns:
            A section dict (see :meth:`_extract`).
        """
        return self._extract(html, "financial_statements")

    def extract_sections(self, html: str) -> dict[str, str]:
        """Extract all target sections that are present in the filing.

        Args:
            html: Raw filing HTML.

        Returns:
            A mapping of section name to extracted text, including only sections
            with non-empty text.
        """
        cleaned = self.clean_html(html)
        found: dict[str, str] = {}
        for name, (item_id, _label, _title) in _SECTION_ITEMS.items():
            body = _extract_item(cleaned, item_id)
            if body:
                found[name] = body
        logger.info("Extracted %d/%d target sections", len(found), len(_SECTION_ITEMS))
        return found
