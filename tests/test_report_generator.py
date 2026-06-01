"""Tests for BaselineReportGenerator."""

from __future__ import annotations

from finsage.evaluation.generators import MockGenerator
from finsage.evaluation.report_generator import BaselineReportGenerator
from finsage.evaluation.runner import EvalRunner


def _results(eval_test_file, tmp_path) -> dict:
    """Run a mock evaluation and return the results dict."""
    runner = EvalRunner(generator=MockGenerator(), output_dir=tmp_path / "figures")
    return runner.run(eval_test_file)


def test_generates_markdown_report(eval_test_file, tmp_path):
    """The report file is written and is non-empty Markdown."""
    results = _results(eval_test_file, tmp_path)
    out = BaselineReportGenerator().generate_markdown_report(
        results, results["paths"]["predictions"], tmp_path / "report.md"
    )
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# FinSage-7B")


def test_report_contains_overall_and_task_metrics(eval_test_file, tmp_path):
    """The report includes overall metrics and per-task sections."""
    results = _results(eval_test_file, tmp_path)
    out = BaselineReportGenerator().generate_markdown_report(
        results, results["paths"]["predictions"], tmp_path / "report.md"
    )
    text = out.read_text(encoding="utf-8")
    assert "## Overall metrics" in text
    assert "## Metrics by task" in text
    assert "filing_qa" in text
    assert "## Task distribution" in text


def test_report_contains_disclaimer(eval_test_file, tmp_path):
    """The report includes the financial disclaimer."""
    results = _results(eval_test_file, tmp_path)
    out = BaselineReportGenerator().generate_markdown_report(
        results, results["paths"]["predictions"], tmp_path / "report.md"
    )
    text = out.read_text(encoding="utf-8")
    assert "## Disclaimer" in text
    assert "not a licensed financial advisor" in text
