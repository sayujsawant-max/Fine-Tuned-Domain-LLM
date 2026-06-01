"""Tests for InstructionBuilder."""

from __future__ import annotations

import json

import pytest

from finsage.data.instruction_builder import TASK_TYPES, InstructionBuilder

SAMPLE_CHUNK = {
    "chunk_id": 42,
    "text": "The company discloses competition and supply chain risks.",
    "metadata": {
        "ticker": "AAPL",
        "year": 2022,
        "filing_type": "10-K",
        "section": "Risk Factors",
    },
}

REQUIRED_KEYS = {"id", "instruction", "input", "output", "task_type", "metadata"}


def test_ten_task_types_defined():
    """There should be exactly ten supported task types."""
    assert len(TASK_TYPES) == 10
    assert len(set(TASK_TYPES)) == 10


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_build_example_for_every_task_type(task_type):
    """Every task type produces a JSONL-serialisable dict with required keys."""
    example = InstructionBuilder().build_example(SAMPLE_CHUNK, task_type)

    assert REQUIRED_KEYS.issubset(example.keys())
    assert example["task_type"] == task_type
    assert example["input"] == SAMPLE_CHUNK["text"]
    assert example["instruction"]  # non-empty
    # Must round-trip through JSON (i.e. be JSONL-ready).
    assert json.loads(json.dumps(example))["task_type"] == task_type


def test_example_id_encodes_metadata():
    """The generated id encodes ticker, year, filing type, and chunk index."""
    example = InstructionBuilder().build_example(SAMPLE_CHUNK, "risk_summary")
    assert example["id"] == "AAPL-2022-10-K-RISK_SUMMARY-0042"


def test_filing_qa_appends_question():
    """For filing_qa, a supplied question is appended to the instruction."""
    example = InstructionBuilder().build_example(
        SAMPLE_CHUNK, "filing_qa", question="What are the risks?"
    )
    assert "What are the risks?" in example["instruction"]


def test_unknown_task_type_raises():
    """An unsupported task type raises ValueError."""
    with pytest.raises(ValueError):
        InstructionBuilder().build_example(SAMPLE_CHUNK, "not_a_task")
