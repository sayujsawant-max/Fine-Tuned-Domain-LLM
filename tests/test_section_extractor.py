"""Tests for SectionExtractor."""

from __future__ import annotations

from finsage.data.section_extractor import SectionExtractor

SAMPLE_HTML = """
<html><body>
<p>Item 1. Business</p>
<p>We make widgets.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business faces competition risk and supply chain risk.</p>
<p>Item 1B. Unresolved Staff Comments</p>
<p>None.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue grew due to strong demand.</p>
<p>Item 7A. Quantitative Disclosures</p>
<p>Item 8. Financial Statements</p>
<p>Net income was reported here.</p>
<p>Item 9. Changes</p>
</body></html>
"""


def test_extract_risk_factors_returns_dict():
    """Risk-factor extraction returns a dict with the expected keys and text."""
    result = SectionExtractor().extract_risk_factors(SAMPLE_HTML)
    assert isinstance(result, dict)
    assert result["section"] == "Risk Factors"
    assert result["item"] == "1A"
    assert "competition risk" in result["text"]
    assert "Item 1B" not in result["text"]


def test_extract_mda_returns_dict():
    """MD&A extraction returns a dict containing the discussion text."""
    result = SectionExtractor().extract_mda(SAMPLE_HTML)
    assert result["section"] == "MD&A"
    assert "Revenue grew" in result["text"]


def test_extract_financial_statements_returns_dict():
    """Financial statement extraction returns a dict containing the body text."""
    result = SectionExtractor().extract_financial_statements(SAMPLE_HTML)
    assert result["section"] == "Financial Statements"
    assert "Net income" in result["text"]


def test_missing_section_returns_empty_text():
    """A document without the section yields an empty (but valid) result."""
    result = SectionExtractor().extract_risk_factors("<html><body>nothing</body></html>")
    assert result["text"] == ""
