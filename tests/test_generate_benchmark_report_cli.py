"""Tests for the generate_benchmark_report CLI."""

from __future__ import annotations

from pathlib import Path

from scripts.generate_benchmark_report import app
from typer.testing import CliRunner

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def test_help_works():
    """`--help` exits cleanly and documents the generate command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.stdout


def test_generate_with_fixture_data(tmp_path):
    """Generate produces a Markdown report and metadata from fixture inputs."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--input-dir",
            str(FIXTURES),
            "--output-dir",
            str(tmp_path),
            "--dataset-stats-path",
            str(FIXTURES / "dataset_stats_sample.json"),
            "--training-summary-path",
            str(FIXTURES / "training_summary_sample.json"),
            "--no-generate-charts",
            "--no-export-pdf",
            "--no-export-html",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "benchmark_report.md").is_file()
    assert (tmp_path / "report_metadata.json").is_file()


def test_strict_mode_fails_when_core_missing(tmp_path):
    """Strict mode exits non-zero when core result files are absent."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app,
        ["generate", "--input-dir", str(empty), "--output-dir", str(tmp_path), "--strict"],
    )
    assert result.exit_code == 1


def test_non_strict_mode_generates_with_warnings(tmp_path):
    """Non-strict mode still produces a report when artifacts are missing."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app,
        [
            "generate",
            "--input-dir",
            str(empty),
            "--output-dir",
            str(tmp_path),
            "--no-generate-charts",
            "--no-export-pdf",
            "--no-export-html",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "benchmark_report.md").is_file()
