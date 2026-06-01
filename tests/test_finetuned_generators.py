"""Tests for fine-tuned generators (no model downloads, no GPU)."""

from __future__ import annotations

import pytest

from finsage.evaluation.generators import (
    AdapterGenerator,
    MergedModelGenerator,
    MockGenerator,
)


def test_mock_finetuned_quality_non_empty():
    """The finetuned-quality mock returns non-empty output."""
    gen = MockGenerator(quality="finetuned")
    out = gen.generate({"task_type": "risk_summary", "input": "Risk one. Risk two. Risk three."})
    assert out.strip()


def test_mock_invalid_quality():
    """An invalid quality value is rejected."""
    with pytest.raises(ValueError):
        MockGenerator(quality="best")


def test_mock_finetuned_more_complete_than_baseline():
    """Finetuned mock returns at least as much as the baseline mock."""
    example = {"task_type": "risk_summary", "input": "One. Two. Three. Four."}
    base = MockGenerator(quality="baseline").generate(example)
    fine = MockGenerator(quality="finetuned").generate(example)
    assert len(fine) >= len(base)


def test_adapter_generator_missing_path_clean_error(tmp_path):
    """AdapterGenerator raises FileNotFoundError for a missing adapter path."""
    with pytest.raises(FileNotFoundError):
        AdapterGenerator(model_id="fake/model", adapter_path=tmp_path / "missing")


def test_adapter_generator_lazy_no_load(tmp_path):
    """Constructing AdapterGenerator with an existing path loads nothing."""
    (tmp_path / "adapter").mkdir()
    gen = AdapterGenerator(model_id="fake/model", adapter_path=tmp_path / "adapter")
    assert gen._model is None  # nothing loaded yet (lazy)
    assert gen.adapter_path.endswith("adapter")


def test_merged_generator_missing_path_clean_error(tmp_path):
    """MergedModelGenerator raises FileNotFoundError for a missing path."""
    with pytest.raises(FileNotFoundError):
        MergedModelGenerator(merged_model_path=tmp_path / "missing")


def test_merged_generator_lazy_no_load(tmp_path):
    """Constructing MergedModelGenerator with an existing path loads nothing."""
    (tmp_path / "merged").mkdir()
    gen = MergedModelGenerator(merged_model_path=tmp_path / "merged")
    assert gen._model is None
    assert gen.name == "merged"
