"""Tests for the benchmark-report chart generation (robust to missing matplotlib)."""

from __future__ import annotations

import json
from pathlib import Path

from finsage.reporting import charts

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_charts_skip_when_matplotlib_unavailable(monkeypatch, tmp_path):
    """When pyplot cannot be imported, chart functions return None (no crash)."""
    monkeypatch.setattr(charts, "_import_pyplot", lambda: None)
    out = charts.create_overall_metrics_chart(_load("comparison_results.json"), tmp_path / "c.png")
    assert out is None
    created = charts.create_all_report_charts(
        {"comparison_results": _load("comparison_results.json")}, tmp_path
    )
    assert created == {}


def test_missing_data_does_not_crash(tmp_path):
    """Empty/None data skips the chart and returns None rather than raising."""
    assert charts.create_overall_metrics_chart({}, tmp_path / "a.png") is None
    assert charts.create_latency_chart(None, tmp_path / "b.png") is None
    assert charts.create_dataset_distribution_chart({}, tmp_path / "c.png") is None
    assert charts.create_hallucination_chart({}, tmp_path / "d.png") is None


def test_charts_created_if_matplotlib_available(tmp_path):
    """If matplotlib is installed, charts are written to disk."""
    if charts._import_pyplot() is None:
        return  # matplotlib not installed in this environment; nothing to assert.
    inputs = {
        "comparison_results": _load("comparison_results.json"),
        "metric_delta_by_task": _load("metric_delta_by_task.json"),
        "dataset_stats": _load("dataset_stats_sample.json"),
        "api_latency": _load("api_latency_benchmark.json"),
    }
    created = charts.create_all_report_charts(inputs, tmp_path)
    assert created  # at least one chart produced
    for path in created.values():
        assert Path(path).is_file()
