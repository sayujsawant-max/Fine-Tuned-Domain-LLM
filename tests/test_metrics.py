"""Tests for the lightweight evaluation metrics."""

from __future__ import annotations

from finsage.evaluation.metrics import (
    compute_exact_match,
    compute_f1,
    compute_rouge_placeholder,
    normalize_text,
)


def test_normalize_text_strips_punct_articles_case():
    """Normalisation lowercases, drops articles, and removes punctuation."""
    assert normalize_text("The Revenue, grew!") == "revenue grew"


def test_exact_match_ignores_punctuation_and_case():
    """Exact match is computed on normalised text."""
    assert compute_exact_match("The Net Income.", "net income")["exact_match"] == 1.0
    assert compute_exact_match("net loss", "net income")["exact_match"] == 0.0


def test_f1_partial_overlap():
    """F1 reflects partial token overlap and stays within [0, 1]."""
    scores = compute_f1("net income increased", "net income decreased")
    assert 0.0 < scores["f1"] < 1.0
    assert scores["precision"] > 0.0
    assert scores["recall"] > 0.0


def test_f1_perfect_and_disjoint():
    """F1 is 1.0 for identical text and 0.0 for disjoint text."""
    assert compute_f1("same words here", "same words here")["f1"] == 1.0
    assert compute_f1("alpha beta", "gamma delta")["f1"] == 0.0


def test_rouge_placeholder_bounds():
    """The ROUGE-L placeholder returns a value within [0, 1]."""
    score = compute_rouge_placeholder(
        "revenue grew due to strong demand",
        "revenue grew because of strong demand",
    )["rouge_l"]
    assert 0.0 < score <= 1.0
