"""Generation backends for baseline evaluation.

Two backends are provided:

- :class:`MockGenerator` — deterministic, dependency-free, used by tests/CI.
- :class:`TransformersGenerator` — real Hugging Face inference; imports ``torch``
  and ``transformers`` lazily so the module loads on a plain CPU machine.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from finsage.evaluation.prompts import (
    build_chat_messages,
    build_eval_prompt,
    normalize_prediction,
)
from finsage.logging_utils import get_logger

logger = get_logger(__name__)

_MISSING_DEPS_MSG = (
    "Real baseline generation requires torch + transformers. "
    "Install training/ml dependencies with pip install -e '.[ml,training]'"
)

_NUMERIC_RE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?%?|\d+(?:\.\d+)?\s?%|\d[\d,]*(?:\.\d+)?\s?"
    r"(?:million|billion|trillion|thousand)|\b\d[\d,]*(?:\.\d+)?\b",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _quantization_config() -> Any:
    """Build a 4-bit NF4 ``BitsAndBytesConfig`` (lazy import).

    Modern transformers removed the ``load_in_4bit=True`` ``from_pretrained``
    kwarg in favour of an explicit ``quantization_config``; this builds it.

    Returns:
        A ``transformers.BitsAndBytesConfig`` for 4-bit NF4 loading.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class BaseGenerator(ABC):
    """Abstract base class for prediction generators."""

    @abstractmethod
    def generate(self, example: dict) -> str:
        """Generate a prediction for a single example.

        Args:
            example: An instruction example.

        Returns:
            The generated (normalised) prediction text.
        """

    def generate_batch(self, examples: list[dict]) -> list[str]:
        """Generate predictions for a batch of examples.

        Args:
            examples: The examples to generate for.

        Returns:
            One prediction per input example, in order.
        """
        return [self.generate(example) for example in examples]


class MockGenerator(BaseGenerator):
    """Deterministic generator for tests and pipeline checks.

    Produces plausible, task-aware answers from the example input without any
    ML dependency, so the evaluation pipeline can be exercised end to end.

    Args:
        quality: ``"baseline"`` or ``"finetuned"``. The ``"finetuned"`` mode
            returns slightly more complete extractive answers (more sentences,
            more metrics) — a deterministic stand-in for an improved model, not
            faked behaviour.
    """

    name = "mock"

    def __init__(self, quality: str = "baseline") -> None:
        if quality not in ("baseline", "finetuned"):
            raise ValueError(f"quality must be 'baseline' or 'finetuned', got {quality!r}")
        self.quality = quality

    def generate(self, example: dict) -> str:
        """Generate a deterministic prediction for an example.

        Args:
            example: An instruction example.

        Returns:
            A normalised, task-appropriate mock prediction.
        """
        task_type = str(example.get("task_type", ""))
        text = str(example.get("input", ""))
        finetuned = self.quality == "finetuned"

        if task_type == "metric_extraction":
            return normalize_prediction(self._metrics_answer(text, limit=8 if finetuned else 5))
        if task_type == "outlook_classification":
            return normalize_prediction("The outlook is neutral based on the excerpt.")
        if task_type == "hallucination_detection":
            return normalize_prediction(self._hallucination_answer(text))
        return normalize_prediction(self._first_sentences(text, count=3 if finetuned else 2))

    @staticmethod
    def _first_sentences(text: str, count: int = 2) -> str:
        """Return the first ``count`` sentences of ``text`` (or a prefix)."""
        sentences = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
        if sentences:
            return " ".join(sentences[:count])
        return text.strip()[:200] or "No content available in the excerpt."

    @staticmethod
    def _metrics_answer(text: str, limit: int = 5) -> str:
        """Return a short metric-style answer if numbers exist in ``text``."""
        values = _NUMERIC_RE.findall(text)
        seen: list[str] = []
        for value in values:
            v = value.strip()
            if v and v not in seen:
                seen.append(v)
        if not seen:
            return "No explicit financial metric was found in the provided excerpt."
        return "Reported metrics: " + ", ".join(seen[:limit]) + "."

    @staticmethod
    def _hallucination_answer(text: str) -> str:
        """Return supported/unsupported based on the proposed-answer marker."""
        lowered = text.lower()
        if "guarantees future investment returns" in lowered or "proposed answer:" not in lowered:
            return "The proposed answer is unsupported by the filing excerpt."
        return "The proposed answer is supported by the filing excerpt."


class TransformersGenerator(BaseGenerator):
    """Real Hugging Face Transformers generator (lazy-loaded).

    Args:
        model_id: Hugging Face model identifier.
        device: Device placement (``"auto"``, ``"cuda"``, ``"cpu"``).
        load_in_4bit: Load the model in 4-bit (requires bitsandbytes + GPU).
        torch_dtype: Torch dtype name or ``"auto"``.
        max_new_tokens: Maximum tokens to generate per example.
        temperature: Sampling temperature; ``0`` selects greedy decoding.
        top_p: Nucleus sampling probability.
        batch_size: Generation batch size.
    """

    name = "transformers"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:
        """Lazily import dependencies and load the model and tokenizer.

        Raises:
            ImportError: If torch/transformers are not installed.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only with deps absent
            raise ImportError(_MISSING_DEPS_MSG) from exc

        self._torch = torch
        if self.device == "cpu" or not torch.cuda.is_available():
            logger.warning(
                "Running the real baseline on CPU is very slow; a GPU is strongly "
                "recommended for Mistral-7B."
            )

        dtype = getattr(torch, self.torch_dtype, None) if self.torch_dtype != "auto" else "auto"
        kwargs: dict = {"device_map": self.device}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.load_in_4bit:
            kwargs["quantization_config"] = _quantization_config()

        logger.info("Loading tokenizer and model for %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def _format_prompt(self, example: dict) -> str:
        """Format the prompt, using the tokenizer chat template if available.

        Args:
            example: An instruction example.

        Returns:
            The formatted prompt string fed to the tokenizer.
        """
        tokenizer = self._tokenizer
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            messages = build_chat_messages(example)
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return build_eval_prompt(example)

    def generate(self, example: dict) -> str:
        """Generate a prediction for one example with the real model.

        Args:
            example: An instruction example.

        Returns:
            The normalised model prediction.
        """
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None and self._torch is not None

        prompt = self._format_prompt(example)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        do_sample = self.temperature > 0.0
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return normalize_prediction(text)


_FINETUNED_DEPS_MSG = (
    "Fine-tuned evaluation requires torch, transformers, and peft. Install "
    "training dependencies with pip install -e '.[ml,training]'"
)


class AdapterGenerator(TransformersGenerator):
    """Evaluate the base model with a trained LoRA adapter (PEFT).

    Args:
        model_id: Base model identifier.
        adapter_path: Path to the trained LoRA adapter directory.
        device: Device placement.
        load_in_4bit: Load the base model in 4-bit.
        torch_dtype: Torch dtype name or ``"auto"``.
        max_new_tokens: Maximum tokens to generate per example.
        temperature: Sampling temperature; ``0`` selects greedy decoding.
        top_p: Nucleus sampling probability.
        batch_size: Generation batch size.

    Raises:
        FileNotFoundError: If ``adapter_path`` does not exist.
    """

    name = "adapter"

    def __init__(
        self,
        model_id: str,
        adapter_path: Path | str,
        device: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        super().__init__(
            model_id=model_id,
            device=device,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        self.adapter_path = str(adapter_path)

    def _ensure_loaded(self) -> None:
        """Load the base model, attach the LoRA adapter, and the tokenizer.

        Raises:
            ImportError: If torch/transformers/peft are not installed.
        """
        if self._model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise ImportError(_FINETUNED_DEPS_MSG) from exc

        self._torch = torch
        if self.device == "cpu" or not torch.cuda.is_available():
            logger.warning("Adapter evaluation on CPU is very slow; a GPU is recommended.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        except (OSError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = getattr(torch, self.torch_dtype, None) if self.torch_dtype != "auto" else "auto"
        kwargs: dict = {"device_map": self.device}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.load_in_4bit:
            kwargs["quantization_config"] = _quantization_config()

        logger.info("Loading base %s + adapter %s", self.model_id, self.adapter_path)
        base = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        model = PeftModel.from_pretrained(base, self.adapter_path)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model


class MergedModelGenerator(TransformersGenerator):
    """Evaluate an already-merged fine-tuned model.

    Args:
        merged_model_path: Path to the merged model directory.
        device: Device placement.
        torch_dtype: Torch dtype name or ``"auto"``.
        max_new_tokens: Maximum tokens to generate per example.
        temperature: Sampling temperature; ``0`` selects greedy decoding.
        top_p: Nucleus sampling probability.
        batch_size: Generation batch size.

    Raises:
        FileNotFoundError: If ``merged_model_path`` does not exist.
    """

    name = "merged"

    def __init__(
        self,
        merged_model_path: Path | str,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        if not Path(merged_model_path).exists():
            raise FileNotFoundError(f"Merged model path not found: {merged_model_path}")
        # The merged model is a standalone CausalLM, so reuse the base loader
        # with the merged path as the model id and no 4-bit quantization.
        super().__init__(
            model_id=str(merged_model_path),
            device=device,
            load_in_4bit=False,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        self.merged_model_path = str(merged_model_path)
