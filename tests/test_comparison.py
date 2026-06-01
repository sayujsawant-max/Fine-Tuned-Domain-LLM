"""Tests for ModelComparison."""

from __future__ import annotations

import json

from finsage.evaluation.comparison import ModelComparison

cmp = ModelComparison()


def test_compare_results_absolute_and_relative_deltas():
    """Absolute and relative deltas are computed per overall metric."""
    base = {"overall": {"token_f1": 0.40}, "by_task": {}}
    fine = {"overall": {"token_f1": 0.60}, "by_task": {}}
    result = cmp.compare_results(base, fine)
    d = result["overall_comparison"]["token_f1"]
    assert d["absolute_delta"] == 0.2
    assert d["relative_delta_pct"] == 50.0
    assert d["improved"] is True


def test_compare_results_detects_improvements_and_regressions():
    """Improvements and regressions are separated."""
    base = {"overall": {"token_f1": 0.40, "rouge_l": 0.50}, "by_task": {}}
    fine = {"overall": {"token_f1": 0.60, "rouge_l": 0.30}, "by_task": {}}
    result = cmp.compare_results(base, fine)
    assert "token_f1" in result["improvements"]
    assert "rouge_l" in result["regressions"]
    assert result["summary"]["metrics_improved"] == 1
    assert result["summary"]["metrics_regressed"] == 1


def test_compare_results_handles_missing_metric():
    """A metric on only one side is compared against 0 and warned about."""
    base = {"overall": {"token_f1": 0.4}, "by_task": {}}
    fine = {"overall": {"token_f1": 0.5, "rouge_l": 0.3}, "by_task": {}}
    result = cmp.compare_results(base, fine)
    assert any("rouge_l" in w for w in result["warnings"])
    assert result["overall_comparison"]["rouge_l"]["baseline"] == 0.0


def test_compare_predictions_joins_by_id(baseline_predictions_file, finetuned_predictions_file):
    """Predictions are joined on id with both predictions present."""
    base = cmp.load_jsonl(baseline_predictions_file)
    fine = cmp.load_jsonl(finetuned_predictions_file)
    rows = cmp.compare_predictions(base, fine, max_examples=10)
    assert len(rows) == 10
    sample = rows[0]
    assert sample["baseline_prediction"] and sample["finetuned_prediction"]
    assert "token_f1" in sample["improvement_summary"]


def test_find_best_improvements_sorted(baseline_predictions_file, finetuned_predictions_file):
    """Best improvements are returned sorted by descending delta."""
    rows = cmp.compare_predictions(
        cmp.load_jsonl(baseline_predictions_file),
        cmp.load_jsonl(finetuned_predictions_file),
        max_examples=10,
    )
    best = cmp.find_best_improvements(rows, metric="token_f1", limit=3)
    deltas = [r["improvement_summary"]["token_f1"] for r in best]
    assert deltas == sorted(deltas, reverse=True)
    assert best[0]["improvement_summary"]["token_f1"] > 0


def test_find_regressions(baseline_predictions_file, finetuned_predictions_file):
    """Regressions (negative token_f1 delta) are detected."""
    rows = cmp.compare_predictions(
        cmp.load_jsonl(baseline_predictions_file),
        cmp.load_jsonl(finetuned_predictions_file),
        max_examples=10,
    )
    regressions = cmp.find_regressions(rows, metric="token_f1", limit=5)
    assert len(regressions) >= 1
    assert all(r["improvement_summary"]["token_f1"] < 0 for r in regressions)


def test_write_comparison_outputs(
    baseline_results_file,
    finetuned_results_file,
    baseline_predictions_file,
    finetuned_predictions_file,
    tmp_path,
):
    """All comparison artifacts are written and are valid JSON/JSONL."""
    comparison = cmp.compare_results(
        cmp.load_json(baseline_results_file), cmp.load_json(finetuned_results_file)
    )
    qualitative = cmp.compare_predictions(
        cmp.load_jsonl(baseline_predictions_file), cmp.load_jsonl(finetuned_predictions_file)
    )
    paths = cmp.write_comparison_outputs(comparison, qualitative, tmp_path)
    for key in (
        "comparison_results",
        "metric_delta_by_task",
        "comparison_summary",
        "qualitative_comparisons",
    ):
        assert paths[key].exists()
    # No NaN/Infinity in the JSON.
    raw = paths["comparison_results"].read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)
