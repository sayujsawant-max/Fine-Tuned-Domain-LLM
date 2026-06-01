"""Loaders for benchmark-report input artifacts.

All loaders use :class:`pathlib.Path`, never crash on a missing *optional* file,
and collect human-readable warnings rather than printing them. A missing
*required* file raises :class:`FileNotFoundError` with a clear message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


def load_json(path: Path | str, required: bool = False) -> dict[str, Any] | None:
    """Load a JSON object from ``path``.

    Args:
        path: Path to the JSON file.
        required: When ``True`` a missing or invalid file raises; otherwise the
            function returns ``None`` and logs a warning.

    Returns:
        The parsed JSON object, or ``None`` when the file is absent/invalid and
        ``required`` is ``False``.

    Raises:
        FileNotFoundError: If ``required`` and the file does not exist.
        ValueError: If ``required`` and the file cannot be parsed as JSON.
    """
    p = Path(path)
    if not p.is_file():
        if required:
            raise FileNotFoundError(f"Required report input not found: {p}")
        logger.warning("Optional report input missing: %s", p)
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        if required:
            raise ValueError(f"Failed to parse required JSON file {p}: {exc}") from exc
        logger.warning("Skipping unreadable JSON file %s: %s", p, exc)
        return None
    if not isinstance(data, dict):
        if required:
            raise ValueError(f"Expected a JSON object in {p}, got {type(data).__name__}")
        logger.warning("Expected JSON object in %s; ignoring.", p)
        return None
    return data


def load_jsonl(path: Path | str, required: bool = False) -> list[dict[str, Any]]:
    """Load a JSON Lines file into a list of objects.

    Args:
        path: Path to the ``.jsonl`` file.
        required: When ``True`` a missing file raises; otherwise an empty list is
            returned and a warning is logged. Malformed individual lines are
            always skipped with a warning (never fatal).

    Returns:
        A list of parsed objects (possibly empty).

    Raises:
        FileNotFoundError: If ``required`` and the file does not exist.
    """
    p = Path(path)
    if not p.is_file():
        if required:
            raise FileNotFoundError(f"Required report input not found: {p}")
        logger.warning("Optional report input missing: %s", p)
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSONL line %d in %s: %s", lineno, p, exc)
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


@dataclass
class ReportInputConfig:
    """Filesystem locations of the artifacts a benchmark report can consume.

    Every path is optional: the report degrades gracefully when a file is
    absent. ``required_keys`` names the logical artifacts that *must* be present
    for the run to be considered a real (non-sample) benchmark; it is consulted
    only in strict mode.

    Attributes:
        input_dir: Directory holding the evaluation/figure artifacts.
        dataset_stats_path: Path to ``dataset_stats.json``.
        validation_report_path: Path to the dataset ``validation_report.json``.
        training_summary_path: Path to the QLoRA ``training_summary.json``.
        baseline_results_path: Path to baseline ``*_results.json``.
        finetuned_results_path: Path to fine-tuned ``*_results.json``.
        comparison_results_path: Path to ``comparison_results.json``.
        comparison_summary_path: Path to ``comparison_summary.json``.
        metric_delta_by_task_path: Path to ``metric_delta_by_task.json``.
        qualitative_path: Path to ``qualitative_comparisons.jsonl``.
        vllm_latency_path: Path to the vLLM latency benchmark JSON.
        api_latency_path: Path to the API latency benchmark JSON.
        deployment_health_path: Path to the full-stack health JSON.
        required_keys: Logical artifact names required in strict mode.
    """

    input_dir: Path = Path("reports/figures")
    dataset_stats_path: Path | None = Path("data/datasets/dataset_stats.json")
    validation_report_path: Path | None = Path("data/datasets/validation_report.json")
    training_summary_path: Path | None = Path("checkpoints/finsage-7b/training_summary.json")
    baseline_results_path: Path | None = None
    finetuned_results_path: Path | None = None
    comparison_results_path: Path | None = None
    comparison_summary_path: Path | None = None
    metric_delta_by_task_path: Path | None = None
    qualitative_path: Path | None = None
    vllm_latency_path: Path | None = None
    api_latency_path: Path | None = None
    deployment_health_path: Path | None = None
    required_keys: tuple[str, ...] = ("comparison_results", "baseline_results", "finetuned_results")

    def __post_init__(self) -> None:
        """Derive default artifact paths from ``input_dir`` when not set."""
        self.input_dir = Path(self.input_dir)
        defaults = {
            "baseline_results_path": "baseline_results.json",
            "finetuned_results_path": "finetuned_results.json",
            "comparison_results_path": "comparison_results.json",
            "comparison_summary_path": "comparison_summary.json",
            "metric_delta_by_task_path": "metric_delta_by_task.json",
            "qualitative_path": "qualitative_comparisons.jsonl",
            "vllm_latency_path": "vllm_latency_benchmark.json",
            "api_latency_path": "api_latency_benchmark.json",
            "deployment_health_path": "full_stack_health.json",
        }
        for attr, filename in defaults.items():
            if getattr(self, attr) is None:
                setattr(self, attr, self.input_dir / filename)
            else:
                setattr(self, attr, Path(getattr(self, attr)))
        for attr in ("dataset_stats_path", "validation_report_path", "training_summary_path"):
            value = getattr(self, attr)
            if value is not None:
                setattr(self, attr, Path(value))


def load_optional_report_inputs(config: ReportInputConfig) -> dict[str, Any]:
    """Load every report input described by ``config``, never crashing on misses.

    Args:
        config: Resolved input-path configuration.

    Returns:
        A dict with one key per logical artifact (value ``None``/``[]`` when
        absent), plus ``"warnings"`` (a list of human-readable strings) and
        ``"missing"`` (the logical names that could not be loaded).
    """
    warnings: list[str] = []
    missing: list[str] = []

    def _json(name: str, path: Path | None) -> dict[str, Any] | None:
        if path is None:
            missing.append(name)
            return None
        data = load_json(path)
        if data is None:
            warnings.append(f"{name}: not available ({path})")
            missing.append(name)
        return data

    inputs: dict[str, Any] = {
        "dataset_stats": _json("dataset_stats", config.dataset_stats_path),
        "validation_report": _json("validation_report", config.validation_report_path),
        "training_summary": _json("training_summary", config.training_summary_path),
        "baseline_results": _json("baseline_results", config.baseline_results_path),
        "finetuned_results": _json("finetuned_results", config.finetuned_results_path),
        "comparison_results": _json("comparison_results", config.comparison_results_path),
        "comparison_summary": _json("comparison_summary", config.comparison_summary_path),
        "metric_delta_by_task": _json("metric_delta_by_task", config.metric_delta_by_task_path),
        "vllm_latency": _json("vllm_latency", config.vllm_latency_path),
        "api_latency": _json("api_latency", config.api_latency_path),
        "deployment_health": _json("deployment_health", config.deployment_health_path),
    }

    qualitative = load_jsonl(config.qualitative_path) if config.qualitative_path else []
    if not qualitative:
        warnings.append(f"qualitative: not available ({config.qualitative_path})")
        missing.append("qualitative")
    inputs["qualitative"] = qualitative

    inputs["warnings"] = warnings
    inputs["missing"] = missing
    return inputs


#: Mapping of logical artifact name -> default location relative to ``base_dir``.
_ARTIFACT_LAYOUT: dict[str, str] = {
    "dataset_stats": "data/datasets/dataset_stats.json",
    "validation_report": "data/datasets/validation_report.json",
    "training_summary": "checkpoints/finsage-7b/training_summary.json",
    "baseline_results": "reports/figures/baseline_results.json",
    "baseline_predictions": "reports/figures/baseline_predictions.jsonl",
    "finetuned_results": "reports/figures/finetuned_results.json",
    "finetuned_predictions": "reports/figures/finetuned_predictions.jsonl",
    "comparison_results": "reports/figures/comparison_results.json",
    "comparison_summary": "reports/figures/comparison_summary.json",
    "metric_delta_by_task": "reports/figures/metric_delta_by_task.json",
    "qualitative_comparisons": "reports/figures/qualitative_comparisons.jsonl",
    "vllm_latency_benchmark": "reports/figures/vllm_latency_benchmark.json",
    "api_latency_benchmark": "reports/figures/api_latency_benchmark.json",
    "deployment_health": "reports/figures/full_stack_health.json",
}


def detect_available_artifacts(base_dir: Path | str = ".") -> dict[str, dict[str, Any]]:
    """Report which known artifacts are present under ``base_dir``.

    Args:
        base_dir: Project root to resolve artifact paths against.

    Returns:
        A mapping of logical artifact name to ``{"path": str, "available": bool}``.
    """
    root = Path(base_dir)
    result: dict[str, dict[str, Any]] = {}
    for name, rel in _ARTIFACT_LAYOUT.items():
        path = root / rel
        result[name] = {"path": str(path), "available": path.is_file()}
    return result
