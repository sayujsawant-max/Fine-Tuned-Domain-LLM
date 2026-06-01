"""Tests for prompt templating."""

from __future__ import annotations

import pytest

from finsage.serving.prompt_templates import (
    build_filing_chat_prompt,
    build_system_prompt,
    validate_task_type,
)

QUESTION = "Summarize the main risk factors."
EXCERPT = "The company faces supply chain disruption and competition."


def test_prompt_includes_grounding_instruction():
    """The system message instructs the model to use only the excerpt."""
    messages = build_filing_chat_prompt(QUESTION, EXCERPT)
    system = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "only the provided filing excerpt" in system
    assert "investment advice" in system


def test_prompt_includes_question_and_excerpt():
    """The user message embeds both the question and the filing excerpt."""
    user = build_filing_chat_prompt(QUESTION, EXCERPT)[1]["content"]
    assert QUESTION in user
    assert EXCERPT in user


def test_prompt_includes_task_type_when_given():
    """A provided task type appears in the user message."""
    user = build_filing_chat_prompt(QUESTION, EXCERPT, task_type="risk_summary")[1]["content"]
    assert "risk_summary" in user


def test_prompt_omits_task_line_when_absent():
    """No 'Task Type:' line is emitted when task_type is None."""
    user = build_filing_chat_prompt(QUESTION, EXCERPT)[1]["content"]
    assert "Task Type:" not in user


def test_invalid_task_type_raises():
    """An unsupported task type raises ValueError."""
    with pytest.raises(ValueError):
        build_filing_chat_prompt(QUESTION, EXCERPT, task_type="bogus")


def test_validate_task_type():
    """validate_task_type passes known types and rejects unknown ones."""
    assert validate_task_type(None) is None
    assert validate_task_type("filing_qa") == "filing_qa"
    with pytest.raises(ValueError):
        validate_task_type("nope")


def test_build_system_prompt_non_empty():
    """build_system_prompt returns a non-empty grounding prompt."""
    assert "FinSage-7B" in build_system_prompt()
