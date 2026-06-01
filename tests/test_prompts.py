"""Tests for evaluation prompt formatting."""

from __future__ import annotations

from finsage.evaluation.prompts import (
    SYSTEM_INSTRUCTION,
    build_chat_messages,
    build_eval_prompt,
    normalize_prediction,
)

EXAMPLE = {
    "task_type": "filing_qa",
    "instruction": "What is the main point of this excerpt?",
    "input": "Revenue grew 8% driven by strong demand.",
}


def test_build_eval_prompt_includes_instruction_and_input():
    """The plain prompt contains the system text, instruction, and input."""
    prompt = build_eval_prompt(EXAMPLE)
    assert SYSTEM_INSTRUCTION in prompt
    assert "What is the main point of this excerpt?" in prompt
    assert "Revenue grew 8% driven by strong demand." in prompt
    assert "filing_qa" in prompt


def test_build_chat_messages_has_system_and_user():
    """Chat messages contain a system and a user role with the content."""
    messages = build_chat_messages(EXAMPLE)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_INSTRUCTION
    assert "Revenue grew 8%" in messages[1]["content"]


def test_normalize_prediction_removes_whitespace_and_tokens():
    """Special tokens and excess whitespace are stripped."""
    raw = "<s>[INST] ignore [/INST]  The answer.\n\n\n   Extra   spaces </s>"
    cleaned = normalize_prediction(raw)
    assert "<s>" not in cleaned
    assert "[/INST]" not in cleaned
    assert "\n\n" not in cleaned
    assert "The answer." in cleaned


def test_normalize_prediction_strips_disclaimer():
    """An injected disclaimer line is removed."""
    raw = "Revenue grew.\nFinSage-7B is not a licensed financial advisor. Verify."
    cleaned = normalize_prediction(raw)
    assert "not a licensed financial advisor" not in cleaned
    assert "Revenue grew." in cleaned


def test_normalize_prediction_handles_empty():
    """Empty input returns an empty string."""
    assert normalize_prediction("") == ""
