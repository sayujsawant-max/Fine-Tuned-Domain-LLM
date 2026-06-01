"""Tests for the report validation logic."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_report import validate_report

from finsage.reporting.report_builder import BenchmarkReportBuilder

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def _generate_valid_report(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a real (fixture-backed) report and return (md, metadata) paths."""
    builder = BenchmarkReportBuilder(
        input_dir=FIXTURES,
        output_dir=tmp_path,
        dataset_stats_path=FIXTURES / "dataset_stats_sample.json",
        training_summary_path=FIXTURES / "training_summary_sample.json",
    )
    outputs = builder.build(
        output_markdown=tmp_path / "benchmark_report.md",
        generate_charts=False,
        export_pdf=False,
        export_html=False,
    )
    return outputs["markdown"], outputs["metadata"]


def test_valid_report_passes(tmp_path):
    """A freshly generated report passes validation."""
    md, meta = _generate_valid_report(tmp_path)
    assert validate_report(md, meta) == []


def test_missing_disclaimer_fails(tmp_path):
    """A report without the disclaimer heading fails validation."""
    md, meta = _generate_valid_report(tmp_path)
    text = md.read_text(encoding="utf-8").replace("## Financial Safety Disclaimer", "## Other")
    md.write_text(text, encoding="utf-8")
    failures = validate_report(md, meta)
    assert any("disclaimer" in f.lower() for f in failures)


def test_missing_limitations_fails(tmp_path):
    """A report without the limitations section fails validation."""
    md, meta = _generate_valid_report(tmp_path)
    text = md.read_text(encoding="utf-8").replace("## Limitations", "## Caveats")
    md.write_text(text, encoding="utf-8")
    failures = validate_report(md, meta)
    assert any("limitations" in f.lower() for f in failures)


def test_todo_detection(tmp_path):
    """A stray TODO marker fails validation unless explicitly allowed."""
    md, meta = _generate_valid_report(tmp_path)
    md.write_text(md.read_text(encoding="utf-8") + "\nTODO: finish this.\n", encoding="utf-8")
    assert any("todo" in f.lower() for f in validate_report(md, meta))
    assert all("todo" not in f.lower() for f in validate_report(md, meta, allow_todo=True))


def test_metadata_validation(tmp_path):
    """A missing metadata file is reported as a failure."""
    md, meta = _generate_valid_report(tmp_path)
    meta.unlink()
    failures = validate_report(md, meta)
    assert any("metadata" in f.lower() for f in failures)


def test_mock_label_required_when_sample(tmp_path):
    """When metadata flags a sample report, the mock label must be present."""
    md, meta = _generate_valid_report(tmp_path)
    # Strip the banner but keep metadata's is_sample_report=True (fixtures are mock).
    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_obj["is_sample_report"] is True
    text = md.read_text(encoding="utf-8").replace(
        "Sample/mock report for pipeline validation only. Not real benchmark results.",
        "(removed)",
    )
    md.write_text(text, encoding="utf-8")
    assert any("mock label" in f.lower() for f in validate_report(md, meta))
