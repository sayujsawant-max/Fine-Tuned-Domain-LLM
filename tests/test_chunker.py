"""Tests for FilingChunker."""

from __future__ import annotations

import pytest

from finsage.data.chunker import FilingChunker


def test_chunk_respects_max_tokens():
    """No chunk should exceed max_tokens worth of whitespace tokens."""
    text = " ".join(str(i) for i in range(100))
    chunks = FilingChunker(max_tokens=20, overlap=5).chunk_text(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(str(chunk["text"]).split()) <= 20


def test_chunk_overlap_behavior():
    """Consecutive chunks should overlap by exactly `overlap` tokens."""
    text = " ".join(str(i) for i in range(50))
    chunker = FilingChunker(max_tokens=10, overlap=3)
    chunks = chunker.chunk_text(text)

    first_end = int(chunks[0]["end_token"])
    second_start = int(chunks[1]["start_token"])
    assert first_end - second_start == 3
    assert chunks[0]["chunk_id"] == 0
    assert chunks[1]["chunk_id"] == 1


def test_chunk_covers_all_tokens():
    """The final chunk should reach the last token of the input."""
    text = " ".join(str(i) for i in range(37))
    chunks = FilingChunker(max_tokens=10, overlap=2).chunk_text(text)
    assert int(chunks[-1]["end_token"]) == 37


def test_empty_text_returns_no_chunks():
    """Blank input yields an empty chunk list."""
    assert FilingChunker().chunk_text("   ") == []


def test_invalid_overlap_raises():
    """Overlap >= max_tokens is rejected at construction."""
    with pytest.raises(ValueError):
        FilingChunker(max_tokens=10, overlap=10)
