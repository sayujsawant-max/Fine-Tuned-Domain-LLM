"""LLM-assisted dataset target generation (Phase "3.5" quality pass).

Rewrites the weak-supervision ``output`` field of SEC-filing instruction
examples into higher-quality, **filing-grounded** answers using the Anthropic
Claude API. The system prompt is frozen and prompt-cached so a batch over many
examples pays the large-prefix cost once. The Anthropic client is injectable for
offline tests, and a deterministic ``mock`` fallback runs with no API at all.

This is optional and gated behind the ``llm`` extra (``pip install -e '.[llm]'``)
plus an ``ANTHROPIC_API_KEY``; it is never imported on the default install path.
"""

from __future__ import annotations

import re
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: Latest Claude model (per the claude-api skill).
DEFAULT_MODEL = "claude-opus-4-8"

_LLM_DEPS_MSG = (
    "LLM-assisted target generation requires the anthropic SDK and an API key. "
    "Install with pip install -e '.[llm]' and set ANTHROPIC_API_KEY."
)

#: Frozen system prompt — stable across all examples so it can be prompt-cached.
SYSTEM_PROMPT = (
    "You are FinSage, a financial-filing analysis expert improving training "
    "labels for a dataset. You are given an instruction, a SEC filing excerpt, "
    "and a weak draft answer. Rewrite the answer so it is accurate, concise, and "
    "grounded ONLY in the provided excerpt.\n\n"
    "Rules:\n"
    "- Use only facts present in the excerpt; never invent figures or claims.\n"
    "- If the excerpt does not support an answer, say so explicitly.\n"
    "- Preserve the answer format implied by the instruction (e.g. a JSON object "
    "for classification/hallucination tasks, a bullet list for extraction).\n"
    "- Do NOT provide investment advice or recommendations.\n"
    "- Respond with ONLY the improved answer — no preamble, no meta-commentary."
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class LLMTargetGenerator:
    """Rewrites instruction-example targets with Claude (or a deterministic mock).

    Args:
        model: Claude model id used for real generation.
        client: An injected Anthropic client (for tests). When ``None`` and not
            in mock mode, a real client is created lazily on first use.
        mock: If ``True``, produce deterministic offline answers and never call
            the API (used by tests and for a no-credentials dry run).
        max_tokens: Maximum tokens to generate per example.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        mock: bool = False,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.mock = mock
        self.max_tokens = max_tokens
        self._client = client

    def _ensure_client(self) -> Any:
        """Return the Anthropic client, creating one lazily if needed.

        Returns:
            An Anthropic client instance.

        Raises:
            ImportError: If the anthropic SDK is not installed.
        """
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise ImportError(_LLM_DEPS_MSG) from exc
        self._client = anthropic.Anthropic()
        return self._client

    def _user_content(self, example: dict[str, Any]) -> str:
        """Build the per-example user message (the volatile, uncached part)."""
        return (
            f"Task type: {example.get('task_type', 'general')}\n"
            f"Instruction: {str(example.get('instruction', '')).strip()}\n\n"
            f"Filing excerpt:\n{str(example.get('input', '')).strip()}\n\n"
            f"Weak draft answer to improve:\n{str(example.get('output', '')).strip()}\n\n"
            "Return only the improved answer."
        )

    def _call_api(self, example: dict[str, Any]) -> str:
        """Generate an improved answer via the Claude API.

        Args:
            example: The instruction example to improve.

        Returns:
            The improved answer text.
        """
        client = self._ensure_client()
        # Frozen system prompt carries the cache breakpoint; the per-example user
        # message (volatile) comes after it so the cached prefix stays stable.
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": self._user_content(example)}],
        )
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        return str(text).strip()

    @staticmethod
    def _mock_answer(example: dict[str, Any]) -> str:
        """Produce a deterministic, offline "improved" answer.

        Slightly richer than the Phase 3 template: the first three meaningful
        sentences of the excerpt, grounded and non-fabricated. Used for tests and
        credential-free dry runs.

        Args:
            example: The instruction example.

        Returns:
            A deterministic improved answer string.
        """
        text = str(example.get("input", "")).strip()
        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if len(s.split()) >= 3]
        if sentences:
            return " ".join(sentences[:3])
        return text[:300] or str(example.get("output", "")).strip()

    def enhance_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``example`` with an improved ``output``.

        Args:
            example: The instruction example to enhance.

        Returns:
            A new example dict with the rewritten ``output`` and metadata flags
            (``generation_method``, ``enhanced``, ``weak_supervision=False``).

        Raises:
            ImportError: If real generation is requested without the anthropic SDK.
        """
        new_output = self._mock_answer(example) if self.mock else self._call_api(example)
        method = "llm_assisted_mock" if self.mock else "llm_assisted"
        metadata = {
            **dict(example.get("metadata", {})),
            "generation_method": method,
            "enhanced": True,
            "weak_supervision": False,
            "enhancement_model": "mock" if self.mock else self.model,
        }
        return {**example, "output": new_output, "metadata": metadata}

    def enhance_examples(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Enhance a list of examples.

        Args:
            examples: The instruction examples to enhance.

        Returns:
            The enhanced examples, in order.
        """
        enhanced: list[dict[str, Any]] = []
        for i, example in enumerate(examples, start=1):
            enhanced.append(self.enhance_example(example))
            if i % 25 == 0:
                logger.info("Enhanced %d/%d examples", i, len(examples))
        return enhanced
