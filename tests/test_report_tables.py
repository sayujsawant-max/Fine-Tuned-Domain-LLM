"""Tests for the benchmark-report table builders."""

from __future__ import annotations

import json
import math
from pathlib import Path

from finsage.reporting.tables import (
    build_dataset_stats_table,
    build_latency_table,
    build_limitations_table,
    build_overall_metrics_table,
    build_task_metrics_table,
    format_delta,
    format_metric,
    markdown_table,
)

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_markdown_table_structure():
    """A table renders a header, separator, and one row per data row."""
    table = markdown_table(["A", "B"], [[1, 2], [3, 4]])
    lines = table.splitlines()
    assert lines[0] == "| A | B |"
    assert lines[1] == "| --- | --- |"
    assert len(lines) == 4


def test_missing_values_become_na():
    """None and non-finite values render as N/A."""
    assert format_metric(None) == "N/A"
    assert format_metric(math.nan) == "N/A"
    assert format_metric(math.inf) == "N/A"
    assert markdown_table(["X"], [[None]]).endswith("| N/A |")


def test_positive_delta_has_plus_sign():
    """Non-negative deltas are prefixed with '+'; negatives keep '-'."""
    assert format_delta(0.2) == "+0.200"
    assert format_delta(0.0) == "+0.000"
    assert format_delta(-0.1) == "-0.100"
    assert format_delta(None) == "N/A"


def test_overall_metrics_table_contains_metrics():
    """The overall table lists compared metrics with deltas."""
    table = build_overall_metrics_table(_load("comparison_results.json"))
    assert "token f1" in table
    assert "+0.220" in table


def test_task_metrics_table_from_delta_by_task():
    """The per-task table accepts a metric_delta_by_task mapping."""
    table = build_task_metrics_table(_load("metric_delta_by_task.json"))
    assert "risk summary" in table
    assert "regressed" in table  # mda_explanation rouge_l regressed


def test_dataset_stats_table_works():
    """Dataset stats render totals and per-task rows."""
    table = build_dataset_stats_table(_load("dataset_stats_sample.json"))
    assert "Total examples" in table
    assert "120" in table


def test_latency_table_works():
    """Latency table renders the percentile rows."""
    table = build_latency_table(_load("api_latency_benchmark.json"))
    assert "p95 latency (s)" in table
    assert "1.430" in table


def test_missing_inputs_degrade_gracefully():
    """None inputs produce a readable 'not available' note, never a crash."""
    assert "not available" in build_overall_metrics_table(None).lower()
    assert "not available" in build_dataset_stats_table(None).lower()
    assert "not available" in build_latency_table(None).lower()


def test_limitations_table_present():
    """The static limitations table always has rows."""
    table = build_limitations_table()
    assert "Limitation" in table
    assert "Not investment advice" in table
