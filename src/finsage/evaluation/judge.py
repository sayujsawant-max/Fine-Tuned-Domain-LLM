"""Optional LLM-as-judge for faithfulness scoring (Phase 7 stub).

An external model (e.g. GPT-4o-mini or Claude) scores whether a generated answer
is faithful to the source filing. This is *optional* and disabled by default in
``configs/eval_config.yaml`` so the core benchmark relies on objective metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LLMJudge:
    """Scores answer faithfulness against source text using an external LLM.

    Args:
        provider: Judge provider identifier (e.g. ``"openai"``).
        model: Model name to use for judging.
        enabled: Whether the judge is active. When ``False``, callers should
            skip judging entirely.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    enabled: bool = False

    def score_faithfulness(self, answer: str, source: str) -> dict[str, float]:
        """Score how faithful ``answer`` is to ``source``.

        Args:
            answer: The generated answer to evaluate.
            source: The filing text the answer should be grounded in.

        Returns:
            A mapping ``{"faithfulness": value}`` in ``[0, 1]``.

        Raises:
            NotImplementedError: Always, until Phase 7 (requires an API key and
                network access).
        """
        raise NotImplementedError("LLM judge scoring lands in Phase 7 and requires an API key.")
