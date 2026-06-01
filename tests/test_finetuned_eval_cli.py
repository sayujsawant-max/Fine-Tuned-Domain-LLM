"""Tests for the fine-tuned evaluation CLI (mock backend, no downloads)."""

from __future__ import annotations

from pathlib import Path

from evaluation.run_finetuned_eval import app
from typer.testing import CliRunner

runner = CliRunner()


def _args(
    test_file: Path | str,
    baseline_results: Path | str,
    baseline_predictions: Path | str,
    out: Path,
    backend: str = "mock",
    extra: list[str] | None = None,
) -> list[str]:
    """Build the CLI argument list."""
    args = [
        "--test-file",
        str(test_file),
        "--baseline-results",
        str(baseline_results),
        "--baseline-predictions",
        str(baseline_predictions),
        "--output-dir",
        str(out),
        "--backend",
        backend,
        "--no-generate-charts",
        "--report-path",
        str(out / "benchmark_report.md"),
    ]
    return args + (extra or [])


def test_mock_backend_generates_outputs(
    eval_test_file, baseline_results_file, baseline_predictions_file, tmp_path
):
    """Mock fine-tuned eval runs and writes all expected outputs."""
    out = tmp_path / "out"
    result = runner.invoke(
        app, _args(eval_test_file, baseline_results_file, baseline_predictions_file, out)
    )
    assert result.exit_code == 0, result.output
    for name in (
        "finetuned_predictions.jsonl",
        "finetuned_results.json",
        "finetuned_metrics_by_task.json",
        "comparison_results.json",
        "metric_delta_by_task.json",
        "comparison_summary.json",
        "qualitative_comparisons.jsonl",
    ):
        assert (out / name).exists(), name
    assert (out / "benchmark_report.md").exists()


def test_missing_baseline_results(eval_test_file, baseline_predictions_file, tmp_path):
    """A missing baseline results file errors cleanly."""
    result = runner.invoke(
        app,
        _args(eval_test_file, tmp_path / "nope.json", baseline_predictions_file, tmp_path / "o"),
    )
    assert result.exit_code == 1
    assert "Baseline results not found" in result.output


def test_missing_baseline_predictions(eval_test_file, baseline_results_file, tmp_path):
    """A missing baseline predictions file errors cleanly."""
    result = runner.invoke(
        app, _args(eval_test_file, baseline_results_file, tmp_path / "nope.jsonl", tmp_path / "o")
    )
    assert result.exit_code == 1
    assert "Baseline predictions not found" in result.output


def test_adapter_backend_missing_adapter_path(
    eval_test_file, baseline_results_file, baseline_predictions_file, tmp_path
):
    """Adapter backend with a missing adapter path errors cleanly."""
    result = runner.invoke(
        app,
        _args(
            eval_test_file,
            baseline_results_file,
            baseline_predictions_file,
            tmp_path / "o",
            backend="adapter",
            extra=["--adapter-path", str(tmp_path / "missing_adapter")],
        ),
    )
    assert result.exit_code == 1
    assert "Adapter path not found" in result.output


def test_merged_backend_missing_path(
    eval_test_file, baseline_results_file, baseline_predictions_file, tmp_path
):
    """Merged backend with a missing model path errors cleanly."""
    result = runner.invoke(
        app,
        _args(
            eval_test_file,
            baseline_results_file,
            baseline_predictions_file,
            tmp_path / "o",
            backend="merged",
            extra=["--merged-model-path", str(tmp_path / "missing_merged")],
        ),
    )
    assert result.exit_code == 1
    assert "Merged model path not found" in result.output
