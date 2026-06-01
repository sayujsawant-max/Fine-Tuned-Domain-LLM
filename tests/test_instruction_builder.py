"""Tests for InstructionBuilder (template/extractive generation)."""

from __future__ import annotations

import json

import pytest

from finsage.data.instruction_builder import TASK_TYPES, InstructionBuilder

builder = InstructionBuilder()

SAMPLE_TEXT = (
    "Net revenue for fiscal 2022 was $394,328 million, an increase of 8% compared with "
    "the prior year. Growth was driven by strong demand and favorable product pricing. "
    "The company faces competition risk and supply chain uncertainty. Gross margin "
    "expanded by 130 basis points to 43.3%."
)


def _chunk(chunk_id: int = 0) -> dict:
    """Build a sample chunk dict."""
    return {
        "chunk_id": chunk_id,
        "text": SAMPLE_TEXT,
        "token_count": len(SAMPLE_TEXT.split()),
        "metadata": {
            "ticker": "AAPL",
            "year": "2022",
            "form": "10-K",
            "section": "mda",
            "accession_number_no_dashes": "000032019322000108",
        },
    }


def test_ten_task_types():
    """There are exactly ten unique task types."""
    assert len(TASK_TYPES) == 10
    assert len(set(TASK_TYPES)) == 10


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_builds_valid_example_for_each_task_type(task_type):
    """Each task type yields a JSON-serialisable example with non-empty fields."""
    example = builder.build_example(_chunk(), task_type)
    for field in ("id", "instruction", "input", "output", "task_type", "source", "metadata"):
        assert field in example
    assert example["task_type"] == task_type
    assert example["instruction"].strip()
    assert len(example["input"]) > 0
    assert len(example["output"]) > 0
    assert example["metadata"]["weak_supervision"] is True
    assert example["metadata"]["generation_method"] == "template_extractive"
    json.loads(json.dumps(example))  # round-trips


def test_rejects_invalid_task_type():
    """Unknown task types raise ValueError."""
    with pytest.raises(ValueError):
        builder.build_example(_chunk(), "not_a_task")
    with pytest.raises(ValueError):
        builder.validate_task_type("bogus")


def test_metric_extraction_finds_money_and_percent():
    """Metric extraction surfaces dollar values and percentages."""
    out = builder.build_example(_chunk(), "metric_extraction")["output"]
    assert "$394,328" in out or "$394,328 million" in out
    assert "8%" in out
    assert "43.3%" in out
    assert "130 basis points" in out


def test_metric_extraction_handles_no_metrics():
    """With no numbers, metric extraction returns the explicit no-metric message."""
    chunk = _chunk()
    chunk["text"] = "The company sells products and provides services to customers."
    out = builder.build_example(chunk, "metric_extraction")["output"]
    assert out == "No explicit financial metric was found in the provided excerpt."


def test_yoy_comparison_extracts_comparison_sentence():
    """YoY extraction captures a sentence with comparison language."""
    out = builder.build_example(_chunk(), "yoy_comparison")["output"]
    assert "compared with" in out or "increase" in out


def test_outlook_classification_returns_valid_label():
    """Outlook classification returns a valid label and a reason."""
    out = json.loads(builder.build_example(_chunk(), "outlook_classification")["output"])
    assert out["label"] in {"positive", "neutral", "negative"}
    assert out["reason"]


def test_hallucination_detection_supported_and_unsupported():
    """Even chunk ids produce supported, odd ids produce unsupported examples."""
    supported = json.loads(builder.build_example(_chunk(0), "hallucination_detection")["output"])
    unsupported = json.loads(builder.build_example(_chunk(1), "hallucination_detection")["output"])
    assert supported["supported"] is True
    assert unsupported["supported"] is False
    # Unsupported input carries the generic, clearly-unsupported claim.
    unsupported_ex = builder.build_example(_chunk(1), "hallucination_detection")
    assert "guarantees future investment returns" in unsupported_ex["input"]


def test_build_examples_for_chunk_respects_selection():
    """build_examples_for_chunk builds one example per requested task type."""
    tasks = ["risk_summary", "filing_qa", "analyst_summary"]
    examples = builder.build_examples_for_chunk(_chunk(), task_types=tasks)
    assert [e["task_type"] for e in examples] == tasks
    assert len({e["id"] for e in examples}) == 3
