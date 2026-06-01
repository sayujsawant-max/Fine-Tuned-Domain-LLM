"""Split long filing text into overlapping, metadata-rich chunks.

Phase 3 uses whitespace tokenization (one token == one whitespace-delimited
word). A real tokenizer can be swapped in later behind the same return shape.
Chunks shorter than ``min_tokens`` are dropped unless they are the only chunk,
and empty chunks are never returned.
"""

from __future__ import annotations

import re
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP = 64
DEFAULT_MIN_TOKENS = 80

_WHITESPACE_RE = re.compile(r"[^\S\n]+")


class FilingChunker:
    """Chunk filing text into overlapping windows of whitespace tokens.

    Args:
        max_tokens: Target maximum number of tokens per chunk.
        overlap: Number of tokens shared between consecutive chunks.
        min_tokens: Minimum tokens for a chunk to be kept (unless it is the
            only chunk produced).

    Raises:
        ValueError: If ``max_tokens`` is not positive, ``overlap`` is negative
            or not smaller than ``max_tokens``, or ``min_tokens`` is negative.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap: int = DEFAULT_OVERLAP,
        min_tokens: int = DEFAULT_MIN_TOKENS,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        if min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.min_tokens = min_tokens

    @staticmethod
    def normalize_text(text: str) -> str:
        """Collapse excessive whitespace and drop empty lines.

        Args:
            text: Raw input text.

        Returns:
            Text with inline whitespace collapsed to single spaces, blank lines
            removed, and surrounding whitespace stripped.
        """
        if not text:
            return ""
        collapsed = _WHITESPACE_RE.sub(" ", text)
        lines = [line.strip() for line in collapsed.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate the token count of ``text`` (whitespace tokenization).

        Args:
            text: The text to measure.

        Returns:
            The number of whitespace-delimited tokens.
        """
        return len(text.split())

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Split ``text`` into overlapping, metadata-carrying chunks.

        Args:
            text: The text to chunk.
            metadata: Optional metadata copied onto every chunk.

        Returns:
            A list of chunk dicts, each with ``chunk_id`` (0-based), ``text``,
            ``token_count``, ``start_token`` (inclusive), ``end_token``
            (exclusive), and ``metadata``. Empty input yields an empty list.
        """
        meta = dict(metadata or {})
        tokens = self.normalize_text(text).split()
        if not tokens:
            return []

        stride = self.max_tokens - self.overlap
        windows: list[tuple[int, int]] = []
        for start in range(0, len(tokens), stride):
            end = min(start + self.max_tokens, len(tokens))
            windows.append((start, end))
            if end == len(tokens):
                break

        # Keep windows of at least min_tokens; keep all if only one window.
        kept = [(s, e) for (s, e) in windows if (e - s) >= self.min_tokens or len(windows) == 1]
        if not kept:  # every window was short — keep the first, non-empty one
            kept = windows[:1]

        chunks: list[dict[str, Any]] = []
        for chunk_id, (start, end) in enumerate(kept):
            chunk_tokens = tokens[start:end]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": " ".join(chunk_tokens),
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "end_token": end,
                    "metadata": dict(meta),
                }
            )

        logger.debug("Chunked %d tokens into %d chunk(s)", len(tokens), len(chunks))
        return chunks
