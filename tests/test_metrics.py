"""Tests for evaluation metrics."""

from __future__ import annotations

from finsage.evaluation.metrics import (
    aggregate_metrics,
    compute_classification_accuracy,
    compute_exact_match,
    compute_lexical_faithfulness,
    compute_metrics_for_example,
    compute_numeric_match,
    compute_rouge_l,
    compute_token_f1,
    extract_numeric_values,
)


def test_exact_match():
    """Exact match ignores punctuation/case/articles."""
    assert compute_exact_match("The Net Income.", "net income")["exact_match"] == 1.0
    assert compute_exact_match("net loss", "net income")["exact_match"] == 0.0


def test_token_f1():
    """Token F1 is 1.0 for identical text and in (0,1) for partial overlap."""
    assert compute_token_f1("net income increased", "net income increased")["token_f1"] == 1.0
    partial = compute_token_f1("net income increased", "net income decreased")["token_f1"]
    assert 0.0 < partial < 1.0


def test_rouge_l():
    """ROUGE-L returns a value within (0, 1] for overlapping text."""
    score = compute_rouge_l("revenue grew on strong demand", "revenue grew due to demand")[
        "rouge_l"
    ]
    assert 0.0 < score <= 1.0
    assert compute_rouge_l("alpha", "beta gamma")["rouge_l"] == 0.0


def test_extract_numeric_values():
    """Numeric extraction finds $, %, magnitudes, and plain numbers."""
    values = extract_numeric_values("Revenue was $81,462 million, up 51%, from 100 units.")
    joined = " ".join(values)
    assert "$81,462" in joined
    assert "51%" in joined
    assert any("million" in v for v in values)
    assert "100" in joined


def test_numeric_match_precision_recall():
    """Numeric match computes exact match, precision, and recall."""
    scores = compute_numeric_match("Revenue $100 million and 5%", "Revenue $100 million and 5%")
    assert scores["numeric_exact_match"] == 1.0
    assert scores["numeric_precision"] == 1.0
    assert scores["numeric_recall"] == 1.0

    partial = compute_numeric_match("$100 million", "$100 million and 5%")
    assert partial["numeric_recall"] < 1.0
    assert partial["numeric_precision"] == 1.0


def test_classification_accuracy_outlook():
    """Outlook labels parse from plain text and JSON."""
    assert (
        compute_classification_accuracy("The outlook is neutral.", '{"label": "neutral"}')[
            "classification_accuracy"
        ]
        == 1.0
    )
    assert (
        compute_classification_accuracy("positive outlook", '{"label": "negative"}')[
            "classification_accuracy"
        ]
        == 0.0
    )


def test_classification_accuracy_hallucination():
    """Supported/unsupported labels parse from text and JSON booleans."""
    assert (
        compute_classification_accuracy(
            "The answer is supported.", '{"supported": true, "reason": "x"}'
        )["classification_accuracy"]
        == 1.0
    )
    assert (
        compute_classification_accuracy(
            "The answer is unsupported.", '{"supported": false, "reason": "x"}'
        )["classification_accuracy"]
        == 1.0
    )
    # Mismatch: predicted supported, reference false.
    assert (
        compute_classification_accuracy("The answer is supported.", '{"supported": false}')[
            "classification_accuracy"
        ]
        == 0.0
    )


def test_lexical_faithfulness_bounds():
    """Lexical faithfulness is within [0, 1] and high for grounded text."""
    source = "Revenue grew due to strong demand in the services segment."
    high = compute_lexical_faithfulness("Revenue grew on strong demand", source)[
        "lexical_faithfulness"
    ]
    low = compute_lexical_faithfulness("Quantum widgets teleport instantly", source)[
        "lexical_faithfulness"
    ]
    assert 0.0 <= low <= high <= 1.0
    assert high > low


def test_compute_metrics_for_example_dispatches_by_task():
    """Metric selection follows the task type."""
    qa = compute_metrics_for_example(
        {"task_type": "filing_qa", "output": "Revenue grew.", "input": "Revenue grew."},
        "Revenue grew.",
    )
    assert "exact_match" in qa and "token_f1" in qa and "lexical_faithfulness" in qa

    metric = compute_metrics_for_example(
        {"task_type": "metric_extraction", "output": "$10 million", "input": "$10 million"},
        "$10 million",
    )
    assert "numeric_exact_match" in metric


def test_aggregate_metrics_overall_and_by_task():
    """Aggregation produces overall and per-task summaries without NaN."""
    rows = [
        {
            "task_type": "filing_qa",
            "prediction": "a b",
            "input_preview": "abc",
            "metrics": {"token_f1": 1.0},
        },
        {
            "task_type": "filing_qa",
            "prediction": "c",
            "input_preview": "de",
            "metrics": {"token_f1": 0.0},
        },
        {
            "task_type": "risk_summary",
            "prediction": "x",
            "input_preview": "f",
            "metrics": {"rouge_l": 0.5},
        },
    ]
    agg = aggregate_metrics(rows)
    assert agg["num_examples"] == 3
    assert agg["overall"]["token_f1"] == 0.5
    assert agg["by_task"]["filing_qa"]["token_f1"] == 0.5
    assert agg["count_by_task"] == {"filing_qa": 2, "risk_summary": 1}
    assert agg["average_input_length"] > 0
