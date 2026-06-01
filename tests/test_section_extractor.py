"""Tests for SectionExtractor using the sample 10-K fixture."""

from __future__ import annotations

from finsage.data.section_extractor import TARGET_SECTIONS, SectionExtractor

extractor = SectionExtractor()


def test_extracts_business(sample_10k_html):
    """Business (Item 1) is extracted and excludes the next section."""
    result = extractor.extract_business(sample_10k_html)
    assert result["section"] == "business"
    assert "premium widgets" in result["text"]
    # Boundary: must not bleed into Risk Factors.
    assert "competition risk" not in result["text"]
    # Must be the real section, not the short table-of-contents entry.
    assert len(result["text"].split()) > 20


def test_extracts_risk_factors(sample_10k_html):
    """Risk Factors (Item 1A) is extracted with its body text."""
    result = extractor.extract_risk_factors(sample_10k_html)
    assert "competition risk" in result["text"]
    assert "supply chain" in result["text"]
    assert "Revenue for fiscal" not in result["text"]


def test_extracts_mda(sample_10k_html):
    """MD&A (Item 7) is extracted with its body text."""
    result = extractor.extract_mda(sample_10k_html)
    assert "Revenue for fiscal 2022 grew 12%" in result["text"]


def test_extracts_market_risk(sample_10k_html):
    """Market Risk (Item 7A) is extracted with its body text."""
    result = extractor.extract_market_risk(sample_10k_html)
    assert "interest-rate risk" in result["text"]


def test_extracts_financial_statements(sample_10k_html):
    """Financial Statements (Item 8) is extracted with its body text."""
    result = extractor.extract_financial_statements(sample_10k_html)
    assert "Net revenue for fiscal 2022 was 1,250" in result["text"]


def test_extract_sections_returns_all_five(sample_10k_html):
    """All five target sections are found in the sample filing."""
    sections = extractor.extract_sections(sample_10k_html)
    assert set(sections) == set(TARGET_SECTIONS)
    assert all(text.strip() for text in sections.values())


def test_handles_missing_section_without_failing():
    """A filing lacking a section yields empty text / no key, not an error."""
    html = "<html><body><p>Nothing relevant here.</p></body></html>"
    assert extractor.extract_risk_factors(html)["text"] == ""
    assert extractor.extract_sections(html) == {}


def test_clean_html_removes_scripts_and_styles(sample_10k_html):
    """Script and style content is stripped from cleaned text."""
    cleaned = extractor.clean_html(sample_10k_html)
    assert "var tracking" not in cleaned
    assert "color: #999" not in cleaned
    assert "ACME WIDGETS" in cleaned


def test_clean_html_has_no_excessive_whitespace(sample_10k_html):
    """Cleaned text has no double spaces or non-breaking spaces."""
    cleaned = extractor.clean_html(sample_10k_html)
    assert "  " not in cleaned
    assert "\xa0" not in cleaned
    assert "\t" not in cleaned
