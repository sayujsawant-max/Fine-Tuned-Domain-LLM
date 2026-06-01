"""Tests for FilingPreprocessor."""

from __future__ import annotations

import json
from pathlib import Path

from finsage.data.preprocessor import FilingPreprocessor

preprocessor = FilingPreprocessor()

REQUIRED_FIELDS = {
    "raw_path",
    "processed_path",
    "section",
    "text_chars",
    "text_words",
    "cik",
    "ticker",
    "form",
    "filing_date",
    "report_date",
    "accession_number",
    "source_url",
}


def _write_raw(tmp_path: Path, html: str, name: str = "filing.html") -> Path:
    """Write raw HTML to a temp path and return it."""
    path = tmp_path / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _metadata(raw_path: Path) -> dict:
    """Build a representative manifest row for a raw filing."""
    return {
        "cik": "0000320193",
        "ticker": "AAPL",
        "form": "10-K",
        "filing_date": "2022-10-28",
        "report_date": "2022-09-24",
        "accession_number": "0000320193-22-000108",
        "accession_number_no_dashes": "000032019322000108",
        "document_url": "https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
        "raw_path": str(raw_path),
    }


def test_process_file_writes_one_txt_per_section(tmp_path, sample_10k_html):
    """One .txt file is written per extracted section."""
    raw = _write_raw(tmp_path, sample_10k_html)
    out_dir = tmp_path / "processed"

    rows = preprocessor.process_file(raw, output_dir=out_dir, metadata=_metadata(raw))

    assert len(rows) == 5
    txt_files = list(out_dir.rglob("*.txt"))
    assert len(txt_files) == 5
    assert all(Path(row["processed_path"]).exists() for row in rows)
    # Path layout: {ticker}/{form}/{year}/{accession}/{section}.txt
    sample = Path(rows[0]["processed_path"])
    assert "AAPL" in sample.parts and "10-K" in sample.parts and "2022" in sample.parts


def test_process_file_returns_required_fields(tmp_path, sample_10k_html):
    """Each returned row carries all required manifest fields."""
    raw = _write_raw(tmp_path, sample_10k_html)
    rows = preprocessor.process_file(raw, output_dir=tmp_path / "p", metadata=_metadata(raw))
    for row in rows:
        assert REQUIRED_FIELDS.issubset(row.keys())


def test_section_word_counts_are_calculated(tmp_path, sample_10k_html):
    """Word and char counts match the written text."""
    raw = _write_raw(tmp_path, sample_10k_html)
    rows = preprocessor.process_file(raw, output_dir=tmp_path / "p", metadata=_metadata(raw))
    for row in rows:
        text = Path(row["processed_path"]).read_text(encoding="utf-8")
        assert row["text_words"] == len(text.split())
        assert row["text_chars"] == len(text)
        assert row["text_words"] > 0


def test_process_manifest_processes_multiple_rows(tmp_path, sample_10k_html):
    """process_manifest handles multiple filings and writes a valid manifest."""
    raw_a = _write_raw(tmp_path, sample_10k_html, "a.html")
    raw_b = _write_raw(tmp_path, sample_10k_html, "b.html")
    manifest_path = tmp_path / "raw_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for raw in (raw_a, raw_b):
            fh.write(json.dumps(_metadata(raw)) + "\n")

    rows = preprocessor.process_manifest(manifest_path, output_dir=tmp_path / "processed")
    assert len(rows) == 10  # 5 sections x 2 filings

    out_manifest = preprocessor.write_manifest(rows, tmp_path / "processed_manifest.jsonl")
    parsed = [json.loads(line) for line in out_manifest.read_text(encoding="utf-8").splitlines()]
    assert len(parsed) == 10
    assert all(REQUIRED_FIELDS.issubset(row.keys()) for row in parsed)
