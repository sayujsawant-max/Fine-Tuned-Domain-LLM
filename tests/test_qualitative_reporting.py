"""Tests for qualitative example selection and rendering."""

from __future__ import annotations

import json
from pathlib import Path

from finsage.reporting.qualitative import (
    build_qualitative_section,
    format_qualitative_example,
    select_qualitative_examples,
)

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURES / "qualitative_comparisons.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def test_examples_selected():
    """Selection returns annotated examples capped at max_examples."""
    chosen = select_qualitative_examples(_rows(), max_examples=4)
    assert 0 < len(chosen) <= 4
    assert all("category" in ex for ex in chosen)


def test_regression_case_included():
    """The mda_explanation row (negative rouge delta) is selectable as regression."""
    chosen = select_qualitative_examples(_rows(), max_examples=5)
    categories = {ex["category"] for ex in chosen}
    assert "regression" in categories


def test_long_text_truncated():
    """Very long fields are truncated with an ellipsis."""
    row = {
        "task_type": "risk_summary",
        "instruction": "Summarize.",
        "input_preview": "x" * 5000,
        "reference": "y" * 5000,
        "baseline_prediction": "z" * 5000,
        "finetuned_prediction": "w" * 5000,
        "improvement_summary": {"rouge_l": 0.1},
        "category": "improvement",
    }
    rendered = format_qualitative_example(row, 1)
    assert "…" in rendered
    assert len(rendered) < 5000  # truncated, not the raw 4x5000 chars


def test_section_includes_both_answers():
    """The rendered section names both base and fine-tuned answers."""
    section = build_qualitative_section(select_qualitative_examples(_rows()))
    assert "Base model:" in section
    assert "Fine-tuned (FinSage-7B):" in section


def test_empty_input_handled():
    """No rows yields a readable note, not a crash."""
    assert "No qualitative examples" in build_qualitative_section([])
    assert select_qualitative_examples([]) == []
