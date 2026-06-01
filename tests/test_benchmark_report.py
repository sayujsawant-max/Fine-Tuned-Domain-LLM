"""Tests for BenchmarkReportGenerator."""

from __future__ import annotations

from finsage.evaluation.comparison import ModelComparison
from finsage.evaluation.report_generator import BenchmarkReportGenerator

cmp = ModelComparison()
gen = BenchmarkReportGenerator()


def _build(
    baseline_results_file,
    finetuned_results_file,
    baseline_predictions_file,
    finetuned_predictions_file,
):
    """Build (comparison, qualitative, base_results, fine_results)."""
    base_results = cmp.load_json(baseline_results_file)
    fine_results = cmp.load_json(finetuned_results_file)
    comparison = cmp.compare_results(base_results, fine_results)
    qualitative = cmp.compare_predictions(
        cmp.load_jsonl(baseline_predictions_file), cmp.load_jsonl(finetuned_predictions_file)
    )
    return comparison, qualitative, base_results, fine_results


def test_report_generated_with_sections(
    baseline_results_file,
    finetuned_results_file,
    baseline_predictions_file,
    finetuned_predictions_file,
    tmp_path,
):
    """The report is written and contains the expected sections."""
    comparison, qualitative, base, fine = _build(
        baseline_results_file,
        finetuned_results_file,
        baseline_predictions_file,
        finetuned_predictions_file,
    )
    out = gen.generate_benchmark_report(base, fine, comparison, qualitative, tmp_path / "report.md")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# FinSage-7B Benchmark Report")
    assert "## Executive summary" in text
    assert "## Overall metrics" in text
    assert "## Metrics by task" in text
    assert "## Qualitative examples" in text
    assert "## Disclaimer" in text
    assert "not a licensed financial advisor" in text


def test_report_flags_mock_when_baseline_is_mock(
    baseline_predictions_file, finetuned_predictions_file, tmp_path
):
    """A mock-backed run is clearly labeled as non-real results."""
    base = {"backend": "mock", "overall": {"token_f1": 0.4}, "by_task": {}, "count_by_task": {}}
    fine = {"backend": "mock", "overall": {"token_f1": 0.5}, "by_task": {}, "count_by_task": {}}
    comparison = cmp.compare_results(base, fine)
    out = gen.generate_benchmark_report(base, fine, comparison, [], tmp_path / "report.md")
    assert "not real benchmark results" in out.read_text(encoding="utf-8")


def test_optional_pdf_export_skips_gracefully(tmp_path):
    """PDF export returns None (or a path) without raising when pandoc is absent."""
    md = tmp_path / "report.md"
    md.write_text("# Report\n", encoding="utf-8")
    result = BenchmarkReportGenerator.optionally_export_pdf(md, tmp_path / "report.pdf")
    assert result is None or result.exists()
