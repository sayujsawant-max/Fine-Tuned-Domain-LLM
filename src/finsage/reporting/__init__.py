"""Benchmark report generation (Phase 11).

This package turns the evaluation, dataset, training, and latency artifacts
produced by earlier phases into a polished, self-contained benchmark report
(Markdown, with optional HTML/PDF exports and charts).

Nothing here requires a GPU, downloads model weights, or calls external
services. When real artifacts are missing the report degrades gracefully and is
clearly labelled as a sample/mock report.
"""

from __future__ import annotations

from finsage.reporting.loaders import (
    ReportInputConfig,
    detect_available_artifacts,
    load_json,
    load_jsonl,
    load_optional_report_inputs,
)
from finsage.reporting.report_builder import BenchmarkReportBuilder

__all__ = [
    "ReportInputConfig",
    "BenchmarkReportBuilder",
    "detect_available_artifacts",
    "load_json",
    "load_jsonl",
    "load_optional_report_inputs",
]
