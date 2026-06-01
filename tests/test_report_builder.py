"""Tests for the BenchmarkReportBuilder."""

from __future__ import annotations

import json
from pathlib import Path

from finsage.reporting.report_builder import MOCK_LABEL, SECTION_TITLES, BenchmarkReportBuilder

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def _builder(output_dir: Path, mock_mode: bool = False) -> BenchmarkReportBuilder:
    return BenchmarkReportBuilder(
        input_dir=FIXTURES,
        output_dir=output_dir,
        dataset_stats_path=FIXTURES / "dataset_stats_sample.json",
        training_summary_path=FIXTURES / "training_summary_sample.json",
        mock_mode=mock_mode,
    )


def test_build_report_context_loads_inputs(tmp_path):
    """Context loads fixture inputs and detects sample (mock) data."""
    context = _builder(tmp_path).build_report_context()
    assert context["inputs"]["comparison_summary"]["metrics_compared"] == 5
    assert context["data_is_sample"] is True  # fixtures use the mock backend
    assert "comparison_results" in context["available_artifacts"]


def test_generate_markdown_includes_required_sections(tmp_path):
    """Every canonical section heading is present in the Markdown."""
    builder = _builder(tmp_path)
    md = builder.generate_markdown(builder.build_report_context())
    for title in SECTION_TITLES:
        assert f"## {title}" in md


def test_markdown_includes_disclaimer_and_limitations(tmp_path):
    """The report carries the disclaimer and limitations content."""
    builder = _builder(tmp_path)
    md = builder.generate_markdown(builder.build_report_context())
    assert "licensed financial advisor" in md.lower()
    assert "## Financial Safety Disclaimer" in md
    assert "## Limitations" in md


def test_mock_label_appears_when_mock_mode(tmp_path):
    """mock_mode=True surfaces the explicit sample/mock banner."""
    builder = _builder(tmp_path, mock_mode=True)
    md = builder.generate_markdown(builder.build_report_context())
    assert MOCK_LABEL in md


def test_build_writes_metadata_and_skips_pdf(tmp_path):
    """build() writes markdown+metadata and tolerates no PDF exporter."""
    builder = _builder(tmp_path)
    outputs = builder.build(
        output_markdown=tmp_path / "benchmark_report.md",
        generate_charts=False,
        export_pdf=False,
        export_html=False,
    )
    assert outputs["markdown"].is_file()
    meta_path = tmp_path / "report_metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["is_sample_report"] is True
    assert "missing_artifacts" in meta
    assert "pdf" not in outputs
