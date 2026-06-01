"""Tests for the benchmark-report input loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from finsage.reporting.loaders import (
    ReportInputConfig,
    detect_available_artifacts,
    load_json,
    load_jsonl,
    load_optional_report_inputs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def test_load_json_works():
    """A present JSON object is parsed into a dict."""
    data = load_json(FIXTURES / "baseline_results.json")
    assert data is not None
    assert data["backend"] == "mock"
    assert "overall" in data


def test_load_jsonl_works():
    """A present JSONL file is parsed into a list of dicts."""
    rows = load_jsonl(FIXTURES / "qualitative_comparisons.jsonl")
    assert len(rows) == 4
    assert rows[0]["task_type"] == "risk_summary"


def test_missing_optional_json_returns_none(tmp_path):
    """A missing optional file returns None and does not raise."""
    assert load_json(tmp_path / "nope.json") is None
    assert load_jsonl(tmp_path / "nope.jsonl") == []


def test_missing_required_file_raises(tmp_path):
    """A missing required file raises a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "nope.json", required=True)
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nope.jsonl", required=True)


def test_load_optional_inputs_collects_warnings():
    """Optional inputs load present files and record warnings for missing ones."""
    config = ReportInputConfig(
        input_dir=FIXTURES,
        dataset_stats_path=FIXTURES / "dataset_stats_sample.json",
        training_summary_path=FIXTURES / "training_summary_sample.json",
    )
    inputs = load_optional_report_inputs(config)
    assert inputs["baseline_results"]["backend"] == "mock"
    assert inputs["dataset_stats"]["total_examples"] == 120
    assert isinstance(inputs["warnings"], list)
    # vllm_latency is absent from the fixtures dir -> recorded as missing.
    assert "vllm_latency" in inputs["missing"]


def test_detect_available_artifacts(tmp_path):
    """detect_available_artifacts flags present vs absent files."""
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    (tmp_path / "reports" / "figures" / "baseline_results.json").write_text("{}", encoding="utf-8")
    detected = detect_available_artifacts(tmp_path)
    assert detected["baseline_results"]["available"] is True
    assert detected["dataset_stats"]["available"] is False
