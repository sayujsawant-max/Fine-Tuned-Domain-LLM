"""Tests for LLMTargetGenerator (mock + injected fake client; no real API)."""

from __future__ import annotations

import pytest

from finsage.data.llm_target_generator import LLMTargetGenerator

EXAMPLE = {
    "id": "ACME-2022-10-K-mda-0-filing_qa",
    "instruction": "What is the main point of this excerpt?",
    "input": "Revenue grew 8% year over year. Margins expanded. Management expects growth.",
    "output": "Revenue grew.",
    "task_type": "filing_qa",
    "metadata": {"ticker": "ACME", "section": "mda", "weak_supervision": True},
}


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, capture: dict) -> None:
        self._capture = capture

    def create(self, **kwargs):
        self._capture.update(kwargs)
        return _Response("IMPROVED filing-grounded answer.")


class _FakeClient:
    def __init__(self) -> None:
        self.captured: dict = {}
        self.messages = _Messages(self.captured)


def test_mock_enhances_and_flags_metadata():
    """Mock mode rewrites output and marks it enhanced (not weak supervision)."""
    out = LLMTargetGenerator(mock=True).enhance_example(EXAMPLE)
    assert out["output"].strip()
    assert out["id"] == EXAMPLE["id"]  # schema preserved
    assert out["metadata"]["enhanced"] is True
    assert out["metadata"]["weak_supervision"] is False
    assert out["metadata"]["generation_method"] == "llm_assisted_mock"
    # Original example is not mutated.
    assert EXAMPLE["metadata"]["weak_supervision"] is True


def test_injected_client_used_with_prompt_cache():
    """Real path uses the injected client and caches the system prompt."""
    fake = _FakeClient()
    gen = LLMTargetGenerator(client=fake, mock=False)
    out = gen.enhance_example(EXAMPLE)

    assert out["output"] == "IMPROVED filing-grounded answer."
    assert out["metadata"]["generation_method"] == "llm_assisted"
    assert out["metadata"]["enhancement_model"] == "claude-opus-4-8"
    # System prompt is sent as a cached block.
    assert fake.captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert fake.captured["model"] == "claude-opus-4-8"


def test_enhance_examples_maps_over_list():
    """enhance_examples returns one enhanced row per input."""
    out = LLMTargetGenerator(mock=True).enhance_examples([EXAMPLE, EXAMPLE])
    assert len(out) == 2
    assert all(row["metadata"]["enhanced"] for row in out)


def test_real_mode_without_anthropic_raises_helpful_error():
    """With no client and anthropic absent, a helpful ImportError is raised."""
    import importlib.util

    if importlib.util.find_spec("anthropic") is not None:
        pytest.skip("anthropic installed; skipping missing-deps assertion")
    with pytest.raises(ImportError, match=r"\.\[llm\]"):
        LLMTargetGenerator(mock=False).enhance_example(EXAMPLE)
