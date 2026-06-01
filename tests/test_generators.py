"""Tests for evaluation generators (no model downloads)."""

from __future__ import annotations

import pytest

from finsage.evaluation.generators import MockGenerator, TransformersGenerator

gen = MockGenerator()


def test_mock_generates_non_empty_answer():
    """The mock generator returns a non-empty answer."""
    out = gen.generate({"task_type": "filing_qa", "input": "Revenue grew 8%. Margins expanded."})
    assert out.strip()


def test_mock_metric_extraction_uses_numbers():
    """For metric_extraction, the mock surfaces numbers from the input."""
    out = gen.generate(
        {"task_type": "metric_extraction", "input": "Revenue was $81,462 million, up 51%."}
    )
    assert "81,462" in out or "51%" in out


def test_mock_outlook_is_neutral_by_default():
    """For outlook_classification, the mock returns neutral."""
    out = gen.generate({"task_type": "outlook_classification", "input": "Mixed signals."})
    assert "neutral" in out.lower()


def test_mock_hallucination_uses_proposed_answer_marker():
    """Hallucination polarity follows the proposed-answer marker."""
    supported = gen.generate(
        {
            "task_type": "hallucination_detection",
            "input": "Revenue grew.\n\nProposed answer: Revenue grew.",
        }
    )
    unsupported = gen.generate(
        {
            "task_type": "hallucination_detection",
            "input": "Revenue grew.\n\nProposed answer: The company guarantees future investment returns.",
        }
    )
    assert "supported" in supported.lower() and "unsupported" not in supported.lower()
    assert "unsupported" in unsupported.lower()


def test_mock_generate_batch_matches_input_count():
    """generate_batch returns one output per input."""
    examples = [
        {"task_type": "filing_qa", "input": "A. B."},
        {"task_type": "risk_summary", "input": "Risk one. Risk two."},
        {"task_type": "metric_extraction", "input": "$10 million and 5%."},
    ]
    outputs = gen.generate_batch(examples)
    assert len(outputs) == len(examples)
    assert all(o.strip() for o in outputs)


def test_transformers_generator_imports_lazily():
    """Constructing the transformers generator does not import torch."""
    generator = TransformersGenerator(model_id="fake/model")
    assert generator.model_id == "fake/model"
    assert generator._model is None  # nothing loaded yet


def test_transformers_generator_raises_helpful_error_without_deps():
    """If torch/transformers are unavailable, a helpful ImportError is raised."""
    pytest.importorskip  # noqa: B018 - documents intent
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        generator = TransformersGenerator(model_id="fake/model")
        with pytest.raises(ImportError, match=r"ml,training"):
            generator.generate({"task_type": "filing_qa", "input": "text"})
    else:
        pytest.skip("torch/transformers installed; skipping missing-deps assertion")
