"""Tests for the report HTML/PDF exporters (robust to missing optional deps)."""

from __future__ import annotations

from pathlib import Path

from finsage.reporting import exporters

_SAMPLE_MD = "# Title\n\n## Section\n\nSome **text** and a list:\n\n- one\n- two\n"


def _write_md(tmp_path: Path) -> Path:
    md = tmp_path / "report.md"
    md.write_text(_SAMPLE_MD, encoding="utf-8")
    return md


def test_check_pdf_export_available_shape():
    """The availability probe returns booleans for each backend."""
    avail = exporters.check_pdf_export_available()
    assert set(avail) == {"markdown", "weasyprint", "reportlab", "pandoc"}
    assert all(isinstance(v, bool) for v in avail.values())


def test_html_export_works_or_falls_back(tmp_path):
    """HTML export always produces a file (markdown package or <pre> fallback)."""
    md = _write_md(tmp_path)
    out = exporters.export_markdown_to_html(md, tmp_path / "report.html")
    assert out is not None and out.is_file()
    html = out.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "Section" in html


def test_pdf_skips_gracefully_and_writes_note(tmp_path, monkeypatch):
    """With no PDF backend, export returns None and writes PDF_EXPORT_SKIPPED.md."""
    monkeypatch.setattr(exporters, "_pdf_via_weasyprint", lambda *a, **k: None)
    monkeypatch.setattr(exporters, "_pdf_via_reportlab", lambda *a, **k: None)
    md = _write_md(tmp_path)
    out = exporters.export_markdown_to_pdf(md, tmp_path / "report.pdf")
    assert out is None
    assert (tmp_path / "PDF_EXPORT_SKIPPED.md").is_file()


def test_pdf_export_missing_source_returns_none(tmp_path):
    """A missing Markdown source yields None without raising."""
    assert exporters.export_markdown_to_pdf(tmp_path / "missing.md", tmp_path / "x.pdf") is None
