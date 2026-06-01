"""Tests for the financial disclaimer helpers."""

from __future__ import annotations

from finsage.serving.disclaimer import (
    FINANCIAL_DISCLAIMER,
    append_disclaimer,
    get_disclaimer,
    remove_duplicate_disclaimer,
)


def test_get_disclaimer_returns_non_empty():
    """get_disclaimer returns the canonical, non-empty disclaimer."""
    assert get_disclaimer() == FINANCIAL_DISCLAIMER
    assert FINANCIAL_DISCLAIMER.strip()


def test_append_disclaimer_adds_disclaimer():
    """append_disclaimer appends the disclaimer to the answer."""
    result = append_disclaimer("Risks include competition.")
    assert result.startswith("Risks include competition.")
    assert FINANCIAL_DISCLAIMER in result


def test_append_disclaimer_does_not_duplicate():
    """Appending twice yields exactly one disclaimer."""
    once = append_disclaimer("Answer.")
    twice = append_disclaimer(once)
    assert twice.count(FINANCIAL_DISCLAIMER) == 1


def test_remove_duplicate_disclaimer_strips_all():
    """remove_duplicate_disclaimer removes embedded disclaimers."""
    text = f"Answer.\n\n{FINANCIAL_DISCLAIMER}\n\n{FINANCIAL_DISCLAIMER}"
    cleaned = remove_duplicate_disclaimer(text)
    assert FINANCIAL_DISCLAIMER not in cleaned
    assert cleaned == "Answer."


def test_append_to_empty_returns_disclaimer():
    """Appending to empty text returns just the disclaimer."""
    assert append_disclaimer("") == FINANCIAL_DISCLAIMER
