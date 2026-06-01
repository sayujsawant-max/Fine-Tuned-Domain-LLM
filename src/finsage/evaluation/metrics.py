"""Lightweight, dependency-free evaluation metrics.

These pure-Python implementations let the evaluation harness and its tests run
on a plain CPU machine. Heavier metrics (real ROUGE-L via ``rouge-score``,
``BERTScore``, NLI-based faithfulness) are added behind the ``ml`` optional
group in later phases — see the TODOs below.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Normalise text for comparison (SQuAD-style).

    Lowercases, removes punctuation and articles, and collapses whitespace.

    Args:
        text: The raw string to normalise.

    Returns:
        The normalised string.
    """
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def compute_exact_match(prediction: str, reference: str) -> dict[str, float]:
    """Compute normalised exact match between a prediction and reference.

    Args:
        prediction: The model's predicted answer.
        reference: The gold reference answer.

    Returns:
        A mapping ``{"exact_match": 1.0 | 0.0}``.
    """
    match = float(normalize_text(prediction) == normalize_text(reference))
    return {"exact_match": match}


def compute_f1(prediction: str, reference: str) -> dict[str, float]:
    """Compute token-level F1 between a prediction and reference.

    Args:
        prediction: The model's predicted answer.
        reference: The gold reference answer.

    Returns:
        A mapping with ``precision``, ``recall``, and ``f1`` keys in ``[0, 1]``.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge_placeholder(prediction: str, reference: str) -> dict[str, float]:
    """Approximate ROUGE-L with a longest-common-subsequence F-measure.

    This is a stand-in so summarisation tasks have a numeric signal in Phase 1.

    TODO(phase-7): replace with ``rouge-score`` (Google) ROUGE-L and add
    BERTScore from the ``ml`` optional dependency group.

    Args:
        prediction: The model's predicted summary.
        reference: The gold reference summary.

    Returns:
        A mapping ``{"rouge_l": value}`` in ``[0, 1]``.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return {"rouge_l": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return {"rouge_l": 0.0}

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    rouge_l = 2 * precision * recall / (precision + recall)
    return {"rouge_l": rouge_l}


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Return the length of the longest common subsequence of two token lists.

    Args:
        a: First token sequence.
        b: Second token sequence.

    Returns:
        The LCS length.
    """
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]
