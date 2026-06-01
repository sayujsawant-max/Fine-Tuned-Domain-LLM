"""Split long filing text into overlapping, metadata-rich chunks.

Phase 1 uses a simple whitespace-token chunker (one token == one whitespace-
delimited word). Phase 3 will swap in a real tokenizer and sentence-boundary
awareness while keeping this same return shape.
"""

from __future__ import annotations

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP = 64


class FilingChunker:
    """Chunk filing text into overlapping windows of whitespace tokens.

    Args:
        max_tokens: Target maximum number of tokens per chunk.
        overlap: Number of tokens shared between consecutive chunks.

    Raises:
        ValueError: If ``max_tokens`` is not positive, or ``overlap`` is
            negative or not strictly smaller than ``max_tokens``.
    """

    def __init__(
        self, max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = DEFAULT_OVERLAP
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk_text(
        self,
        text: str,
        max_tokens: int | None = None,
        overlap: int | None = None,
    ) -> list[dict[str, object]]:
        """Split ``text`` into overlapping chunks.

        Args:
            text: The text to chunk.
            max_tokens: Override the instance ``max_tokens`` for this call.
            overlap: Override the instance ``overlap`` for this call.

        Returns:
            A list of dicts, each with ``chunk_id`` (0-based index), ``text``,
            ``start_token`` (inclusive), and ``end_token`` (exclusive). Returns
            an empty list for blank input.

        Raises:
            ValueError: If the effective ``overlap`` is not smaller than the
                effective ``max_tokens``.
        """
        size = max_tokens or self.max_tokens
        step_overlap = self.overlap if overlap is None else overlap
        if step_overlap >= size:
            raise ValueError("overlap must be smaller than max_tokens")

        tokens = text.split()
        if not tokens:
            return []

        stride = size - step_overlap
        chunks: list[dict[str, object]] = []
        for chunk_id, start in enumerate(range(0, len(tokens), stride)):
            end = min(start + size, len(tokens))
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": " ".join(tokens[start:end]),
                    "start_token": start,
                    "end_token": end,
                }
            )
            if end == len(tokens):
                break

        logger.debug("Chunked %d tokens into %d chunks", len(tokens), len(chunks))
        return chunks
