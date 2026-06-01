"""Evaluation metrics for FinSage baseline scoring.

Pure-Python and dependency-free so the harness and tests run on a plain CPU
machine. Includes exact match, token F1, a hand-rolled ROUGE-L, numeric
extraction/matching, classification-label parsing, and a lightweight lexical
faithfulness proxy. All functions return JSON-serialisable floats and never
produce NaN.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

#: Numeric patterns: money, percentages, magnitudes, and plain numbers.
_NUMERIC_PATTERNS = (
    re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\d+(?:\.\d+)?\s?%"),
    re.compile(r"\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion|thousand)", re.IGNORECASE),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\b"),
)

_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "as",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "their",
        "our",
        "your",
        "his",
        "her",
        "they",
        "we",
        "you",
        "i",
        "he",
        "she",
        "them",
        "us",
        "not",
        "no",
        "but",
        "if",
        "then",
        "than",
        "into",
        "over",
        "under",
        "about",
        "above",
        "below",
        "up",
        "down",
        "out",
    ]
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalise text SQuAD-style (lowercase, drop punctuation + articles).

    Args:
        text: The raw string.

    Returns:
        The normalised string.
    """
    text = text.lower().translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_text_for_eval(text: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace.

    Args:
        text: The raw string.

    Returns:
        The normalised string (articles retained, unlike :func:`normalize_text`).
    """
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


# ---------------------------------------------------------------------------
# Text-overlap metrics
# ---------------------------------------------------------------------------
def compute_exact_match(prediction: str, reference: str) -> dict[str, float]:
    """Compute normalised exact match.

    Args:
        prediction: The predicted answer.
        reference: The gold answer.

    Returns:
        ``{"exact_match": 1.0 | 0.0}``.
    """
    return {"exact_match": float(normalize_text(prediction) == normalize_text(reference))}


def compute_f1(prediction: str, reference: str) -> dict[str, float]:
    """Compute token-level precision/recall/F1 (back-compatible helper).

    Args:
        prediction: The predicted answer.
        reference: The gold answer.

    Returns:
        ``{"precision", "recall", "f1"}`` in ``[0, 1]``.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def compute_token_f1(prediction: str, reference: str) -> dict[str, float]:
    """Compute token F1 as a single ``token_f1`` metric.

    Args:
        prediction: The predicted answer.
        reference: The gold answer.

    Returns:
        ``{"token_f1": value}`` in ``[0, 1]``.
    """
    return {"token_f1": compute_f1(prediction, reference)["f1"]}


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Return the longest-common-subsequence length of two token lists.

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
            curr[j] = prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def compute_rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """Compute a lightweight ROUGE-L (LCS-based F-measure).

    Args:
        prediction: The predicted text.
        reference: The gold text.

    Returns:
        ``{"rouge_l": value}`` in ``[0, 1]``.
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
    return {"rouge_l": 2 * precision * recall / (precision + recall)}


def compute_rouge_placeholder(prediction: str, reference: str) -> dict[str, float]:
    """Backwards-compatible alias for :func:`compute_rouge_l`.

    Args:
        prediction: The predicted text.
        reference: The gold text.

    Returns:
        ``{"rouge_l": value}``.
    """
    return compute_rouge_l(prediction, reference)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _parse_label(text: str) -> str:
    """Parse a classification label from text or a JSON object.

    Recognises outlook labels (positive/neutral/negative) and hallucination
    labels (supported/unsupported), preferring an explicit JSON payload.

    Args:
        text: The prediction or reference text.

    Returns:
        One of ``positive``, ``neutral``, ``negative``, ``supported``,
        ``unsupported``, or ``unknown``.
    """
    stripped = (text or "").strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            if "label" in obj:
                return str(obj["label"]).strip().lower()
            if "supported" in obj:
                return "supported" if bool(obj["supported"]) else "unsupported"
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    lowered = stripped.lower()
    if "unsupported" in lowered or "not supported" in lowered:
        return "unsupported"
    if "supported" in lowered:
        return "supported"
    positions = {label: lowered.find(label) for label in ("positive", "neutral", "negative")}
    positions = {label: pos for label, pos in positions.items() if pos >= 0}
    if positions:
        return min(positions, key=lambda label: positions[label])
    return "unknown"


def compute_classification_accuracy(prediction: str, reference: str) -> dict[str, float]:
    """Compute label-match accuracy for classification-style tasks.

    Args:
        prediction: The predicted answer.
        reference: The gold answer.

    Returns:
        ``{"classification_accuracy": 1.0 | 0.0}`` (0.0 if either label is
        unknown).
    """
    pred_label = _parse_label(prediction)
    ref_label = _parse_label(reference)
    if pred_label == "unknown" or ref_label == "unknown":
        return {"classification_accuracy": 0.0}
    return {"classification_accuracy": float(pred_label == ref_label)}


# ---------------------------------------------------------------------------
# Numeric metrics
# ---------------------------------------------------------------------------
def extract_numeric_values(text: str) -> list[str]:
    """Extract dollar values, percentages, magnitudes, and plain numbers.

    Args:
        text: Source text.

    Returns:
        A de-duplicated list of matched value strings, in first-seen order.
    """
    found: list[str] = []
    for pattern in _NUMERIC_PATTERNS:
        for match in pattern.findall(text or ""):
            value = match.strip()
            if value and value not in found:
                found.append(value)
    return found


def _normalize_numeric(value: str) -> str:
    """Canonicalise a numeric token for comparison (strip ``$``/commas/spaces)."""
    return value.lower().replace("$", "").replace(",", "").replace(" ", "")


def compute_numeric_match(prediction: str, reference: str) -> dict[str, float]:
    """Compare numeric values between a prediction and reference.

    Args:
        prediction: The predicted answer.
        reference: The gold answer.

    Returns:
        ``{"numeric_exact_match", "numeric_precision", "numeric_recall"}``.
    """
    pred = {_normalize_numeric(v) for v in extract_numeric_values(prediction)}
    ref = {_normalize_numeric(v) for v in extract_numeric_values(reference)}

    if not pred and not ref:
        return {"numeric_exact_match": 1.0, "numeric_precision": 1.0, "numeric_recall": 1.0}

    true_positive = len(pred & ref)
    precision = true_positive / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = true_positive / len(ref) if ref else 1.0
    return {
        "numeric_exact_match": float(pred == ref),
        "numeric_precision": precision,
        "numeric_recall": recall,
    }


# ---------------------------------------------------------------------------
# Faithfulness (lightweight proxy)
# ---------------------------------------------------------------------------
def compute_lexical_faithfulness(prediction: str, source_text: str) -> dict[str, float]:
    """Estimate faithfulness as the share of prediction content words in source.

    This is a lexical proxy, **not** a true NLI faithfulness metric — it is a
    cheap baseline placeholder replaced by NLI/LLM-judge in later phases.

    Args:
        prediction: The predicted answer.
        source_text: The source filing excerpt the answer should be grounded in.

    Returns:
        ``{"lexical_faithfulness": value}`` in ``[0, 1]``.
    """
    pred_words = {w for w in normalize_text_for_eval(prediction).split() if w not in _STOPWORDS}
    if not pred_words:
        return {"lexical_faithfulness": 1.0}
    source_words = set(normalize_text_for_eval(source_text).split())
    grounded = len(pred_words & source_words)
    return {"lexical_faithfulness": grounded / len(pred_words)}


# ---------------------------------------------------------------------------
# Dispatch + aggregation
# ---------------------------------------------------------------------------
_TASK_METRICS: dict[str, tuple[str, ...]] = {
    "filing_qa": ("exact_match", "token_f1", "lexical_faithfulness"),
    "risk_summary": ("rouge_l", "token_f1", "lexical_faithfulness"),
    "mda_explanation": ("rouge_l", "token_f1", "lexical_faithfulness"),
    "metric_extraction": ("numeric_match", "token_f1", "lexical_faithfulness"),
    "yoy_comparison": ("rouge_l", "token_f1", "lexical_faithfulness"),
    "business_risk_identification": ("rouge_l", "token_f1", "lexical_faithfulness"),
    "revenue_driver_explanation": ("rouge_l", "token_f1", "lexical_faithfulness"),
    "analyst_summary": ("rouge_l", "token_f1", "lexical_faithfulness"),
    "outlook_classification": ("classification_accuracy", "token_f1"),
    "hallucination_detection": ("classification_accuracy", "lexical_faithfulness"),
}

_DEFAULT_METRICS = ("token_f1", "lexical_faithfulness")


def compute_metrics_for_example(example: dict, prediction: str) -> dict[str, float]:
    """Compute the metrics appropriate to an example's task type.

    Args:
        example: An instruction example with ``task_type``, ``output``, ``input``.
        prediction: The model prediction.

    Returns:
        A flat mapping of metric name to value (always JSON-serialisable).
    """
    task_type = str(example.get("task_type", ""))
    reference = str(example.get("output", ""))
    source = str(example.get("input", ""))
    metric_names = _TASK_METRICS.get(task_type, _DEFAULT_METRICS)

    result: dict[str, float] = {}
    for name in metric_names:
        try:
            if name == "exact_match":
                result.update(compute_exact_match(prediction, reference))
            elif name == "token_f1":
                result.update(compute_token_f1(prediction, reference))
            elif name == "rouge_l":
                result.update(compute_rouge_l(prediction, reference))
            elif name == "numeric_match":
                result.update(compute_numeric_match(prediction, reference))
            elif name == "classification_accuracy":
                result.update(compute_classification_accuracy(prediction, reference))
            elif name == "lexical_faithfulness":
                result.update(compute_lexical_faithfulness(prediction, source))
        except Exception:  # pragma: no cover - defensive, must never crash eval
            logger.warning("Metric %s failed for task %s; recording 0.0", name, task_type)
            result[name] = 0.0
    return result


def _mean(values: list[float]) -> float:
    """Return the mean of ``values`` (0.0 for an empty list)."""
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate_metrics(rows: list[dict]) -> dict[str, Any]:
    """Aggregate per-example metrics into overall and per-task summaries.

    Args:
        rows: Result rows, each with a ``metrics`` dict, ``task_type``,
            ``input_preview``/``input``, and ``prediction``.

    Returns:
        A dict with ``overall`` metrics, ``by_task`` metrics, ``count_by_task``,
        ``average_input_length``, ``average_prediction_length``, and
        ``num_examples``.
    """
    overall: dict[str, list[float]] = {}
    by_task: dict[str, dict[str, list[float]]] = {}
    count_by_task: Counter[str] = Counter()
    input_lengths: list[float] = []
    pred_lengths: list[float] = []

    for row in rows:
        task = str(row.get("task_type", "unknown"))
        count_by_task[task] += 1
        input_lengths.append(float(len(str(row.get("input_preview", row.get("input", ""))))))
        pred_lengths.append(float(len(str(row.get("prediction", "")))))
        for name, value in (row.get("metrics") or {}).items():
            fvalue = float(value)
            overall.setdefault(name, []).append(fvalue)
            by_task.setdefault(task, {}).setdefault(name, []).append(fvalue)

    return {
        "num_examples": len(rows),
        "overall": {name: _mean(values) for name, values in sorted(overall.items())},
        "by_task": {
            task: {name: _mean(values) for name, values in sorted(metrics.items())}
            for task, metrics in sorted(by_task.items())
        },
        "count_by_task": dict(sorted(count_by_task.items())),
        "average_input_length": _mean(input_lengths),
        "average_prediction_length": _mean(pred_lengths),
    }
