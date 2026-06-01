"""Tests for the Pydantic request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from finsage.serving.schemas import ChatRequest, OpenAIChatCompletionRequest


def test_valid_chat_request():
    """A well-formed ChatRequest validates and keeps defaults."""
    req = ChatRequest(question="What are the risks?", filing_excerpt="Supply chain risk.")
    assert req.max_tokens == 256
    assert req.temperature == 0.0
    assert req.include_disclaimer is True


def test_empty_question_fails():
    """A blank question is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(question="   ", filing_excerpt="text")


def test_empty_filing_excerpt_fails():
    """A blank filing excerpt is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(question="q", filing_excerpt="")


@pytest.mark.parametrize("max_tokens", [0, -5, 4096])
def test_invalid_max_tokens_fails(max_tokens):
    """max_tokens outside 1..2048 is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(question="q", filing_excerpt="e", max_tokens=max_tokens)


@pytest.mark.parametrize("temperature", [-0.1, 2.5])
def test_invalid_temperature_fails(temperature):
    """temperature outside 0.0..2.0 is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(question="q", filing_excerpt="e", temperature=temperature)


def test_invalid_task_type_fails():
    """An unknown task_type is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(question="q", filing_excerpt="e", task_type="not_a_task")


def test_valid_task_type_passes():
    """A supported task_type validates."""
    req = ChatRequest(question="q", filing_excerpt="e", task_type="risk_summary")
    assert req.task_type == "risk_summary"


def test_openai_request_requires_messages():
    """An empty messages list is rejected."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionRequest(messages=[])


def test_openai_request_validates_messages():
    """A valid OpenAI request parses messages into models."""
    req = OpenAIChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=10)
    assert req.messages[0].role == "user"
    assert req.stream is False
