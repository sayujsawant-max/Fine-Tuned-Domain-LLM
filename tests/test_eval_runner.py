"""Tests for EvalRunner with the mock generator."""

from __future__ import annotations

import json

from finsage.evaluation.generators import MockGenerator
from finsage.evaluation.runner import EvalRunner


def _runner(tmp_path) -> EvalRunner:
    """Build an EvalRunner with a mock generator and a temp output dir."""
    return EvalRunner(generator=MockGenerator(), output_dir=tmp_path / "figures", save_every=5)


def test_loads_examples(eval_test_file, tmp_path):
    """The runner loads JSONL examples and respects max_examples."""
    runner = _runner(tmp_path)
    assert len(runner.load_examples(eval_test_file)) == 10
    assert len(runner.load_examples(eval_test_file, max_examples=3)) == 3


def test_run_writes_outputs(eval_test_file, tmp_path):
    """A full run writes predictions and metrics files and returns results."""
    runner = _runner(tmp_path)
    results = runner.run(eval_test_file)

    figures = tmp_path / "figures"
    assert (figures / "baseline_predictions.jsonl").exists()
    assert (figures / "baseline_results.json").exists()
    assert (figures / "baseline_metrics_by_task.json").exists()

    assert results["num_examples"] == 10
    assert results["backend"] == "mock"
    assert "overall" in results and "by_task" in results


def test_predictions_have_prediction_and_metrics(eval_test_file, tmp_path):
    """Each prediction row carries a prediction and a metrics dict."""
    runner = _runner(tmp_path)
    runner.run(eval_test_file)
    lines = (
        (tmp_path / "figures" / "baseline_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 10
    for line in lines:
        row = json.loads(line)
        assert row["prediction"]
        assert isinstance(row["metrics"], dict) and row["metrics"]
        assert "task_type" in row and "reference" in row


def test_metrics_json_is_serializable_no_nan(eval_test_file, tmp_path):
    """The results JSON parses and contains no NaN values."""
    runner = _runner(tmp_path)
    runner.run(eval_test_file)
    raw = (tmp_path / "figures" / "baseline_results.json").read_text(encoding="utf-8")
    assert "NaN" not in raw
    data = json.loads(raw)
    assert data["count_by_task"]  # all 10 task types present
    assert len(data["count_by_task"]) == 10


def test_max_examples_limit(eval_test_file, tmp_path):
    """max_examples limits the number of evaluated examples."""
    runner = _runner(tmp_path)
    results = runner.run(eval_test_file, max_examples=4)
    assert results["num_examples"] == 4
