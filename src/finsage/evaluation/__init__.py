"""Evaluation: prompts, generators, metrics, runner, and report generation."""

from __future__ import annotations

from finsage.evaluation.comparison import ModelComparison
from finsage.evaluation.generators import (
    AdapterGenerator,
    BaseGenerator,
    MergedModelGenerator,
    MockGenerator,
    TransformersGenerator,
)
from finsage.evaluation.metrics import (
    aggregate_metrics,
    build_nli_scorer,
    compute_classification_accuracy,
    compute_exact_match,
    compute_f1,
    compute_lexical_faithfulness,
    compute_metrics_for_example,
    compute_nli_faithfulness,
    compute_numeric_match,
    compute_rouge_l,
    compute_rouge_placeholder,
    compute_token_f1,
    extract_numeric_values,
)
from finsage.evaluation.prompts import (
    build_chat_messages,
    build_eval_prompt,
    normalize_prediction,
)
from finsage.evaluation.report_generator import (
    BaselineReportGenerator,
    BenchmarkReportGenerator,
)
from finsage.evaluation.runner import EvalRunner

__all__ = [
    "AdapterGenerator",
    "BaseGenerator",
    "BaselineReportGenerator",
    "BenchmarkReportGenerator",
    "EvalRunner",
    "MergedModelGenerator",
    "MockGenerator",
    "ModelComparison",
    "TransformersGenerator",
    "aggregate_metrics",
    "build_chat_messages",
    "build_eval_prompt",
    "build_nli_scorer",
    "compute_classification_accuracy",
    "compute_exact_match",
    "compute_f1",
    "compute_lexical_faithfulness",
    "compute_metrics_for_example",
    "compute_nli_faithfulness",
    "compute_numeric_match",
    "compute_rouge_l",
    "compute_rouge_placeholder",
    "compute_token_f1",
    "extract_numeric_values",
    "normalize_prediction",
]
