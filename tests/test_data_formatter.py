"""Tests for the SFT data formatter."""

from __future__ import annotations

from finsage.training.data_formatter import (
    count_tokens_approx,
    format_dataset_for_sft,
    format_sft_example,
    validate_training_example,
)

EXAMPLE = {
    "task_type": "filing_qa",
    "instruction": "What is the main point?",
    "input": "Revenue grew 8% on strong demand.",
    "output": "Revenue grew 8% on strong demand.",
}


def test_format_includes_all_fields():
    """The formatted SFT text includes task type, instruction, input, output."""
    text = format_sft_example(EXAMPLE)
    assert "filing_qa" in text
    assert "What is the main point?" in text
    assert "Revenue grew 8% on strong demand." in text
    assert "[INST]" in text and "[/INST]" in text
    assert "Do not provide investment advice." in text


def test_validate_passes_valid_example():
    """A complete example yields no validation errors."""
    assert validate_training_example(EXAMPLE) == []


def test_validate_catches_missing_and_empty_fields():
    """Missing and empty required fields are reported."""
    missing = {"task_type": "filing_qa", "instruction": "x", "input": "y"}
    errors = validate_training_example(missing)
    assert any("output" in e for e in errors)

    empty = {**EXAMPLE, "instruction": "   "}
    assert any("instruction" in e for e in validate_training_example(empty))


def test_count_tokens_approx_positive():
    """Token approximation returns a positive count for non-empty text."""
    assert count_tokens_approx("one two three") == 3
    assert count_tokens_approx(format_sft_example(EXAMPLE)) > 0


def test_format_dataset_for_sft_list():
    """Formatting a list of examples adds a 'text' field to each."""
    out = format_dataset_for_sft([EXAMPLE, EXAMPLE])
    assert len(out) == 2
    assert all("text" in row and row["text"] for row in out)
