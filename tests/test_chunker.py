"""Tests for FilingChunker."""

from __future__ import annotations

import pytest

from finsage.data.chunker import FilingChunker


def _numbered(n: int) -> str:
    """Return ``n`` whitespace-separated tokens."""
    return " ".join(str(i) for i in range(n))


def test_chunks_long_text_into_multiple_chunks():
    """Long text is split into more than one chunk."""
    chunks = FilingChunker(max_tokens=50, overlap=10, min_tokens=5).chunk_text(_numbered(200))
    assert len(chunks) > 1
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_respects_max_tokens():
    """No chunk exceeds max_tokens."""
    chunks = FilingChunker(max_tokens=40, overlap=8, min_tokens=5).chunk_text(_numbered(200))
    assert all(c["token_count"] <= 40 for c in chunks)


def test_respects_overlap():
    """Consecutive chunks overlap by exactly `overlap` tokens."""
    chunks = FilingChunker(max_tokens=30, overlap=7, min_tokens=5).chunk_text(_numbered(120))
    assert chunks[0]["end_token"] - chunks[1]["start_token"] == 7


def test_drops_short_trailing_chunk():
    """A trailing window shorter than min_tokens is dropped when others exist."""
    # 110 tokens, stride 50 -> windows [0,60],[50,110]? compute: max=60 overlap=10 stride=50
    chunker = FilingChunker(max_tokens=60, overlap=10, min_tokens=40)
    chunks = chunker.chunk_text(_numbered(115))
    # last window would be tokens[100:115] = 15 tokens (< 40) and must be dropped
    assert all(c["token_count"] >= 40 for c in chunks)


def test_keeps_only_chunk_even_if_short():
    """A single short chunk is kept (never returns empty)."""
    chunks = FilingChunker(max_tokens=512, overlap=64, min_tokens=80).chunk_text(_numbered(10))
    assert len(chunks) == 1
    assert chunks[0]["token_count"] == 10


def test_preserves_metadata():
    """Metadata is copied onto every chunk."""
    meta = {"ticker": "AAPL", "section": "mda"}
    chunks = FilingChunker(max_tokens=20, overlap=5, min_tokens=3).chunk_text(_numbered(60), meta)
    assert all(c["metadata"] == meta for c in chunks)
    # Defensive copy: mutating a chunk's metadata must not affect the source.
    chunks[0]["metadata"]["ticker"] = "MUT"
    assert meta["ticker"] == "AAPL"


def test_normalizes_whitespace():
    """Excessive whitespace and blank lines are collapsed."""
    text = "alpha   beta\n\n\n  gamma\t\tdelta   "
    chunks = FilingChunker(max_tokens=10, overlap=2, min_tokens=1).chunk_text(text)
    assert chunks[0]["text"] == "alpha beta gamma delta"


def test_empty_text_returns_no_chunks():
    """Blank input yields an empty list, never an empty chunk."""
    assert FilingChunker().chunk_text("   \n\n  ") == []


def test_estimate_tokens():
    """Token estimation uses whitespace splitting."""
    assert FilingChunker.estimate_tokens("one two three") == 3
    assert FilingChunker.estimate_tokens("") == 0


def test_invalid_overlap_raises():
    """Overlap >= max_tokens is rejected at construction."""
    with pytest.raises(ValueError):
        FilingChunker(max_tokens=10, overlap=10)
