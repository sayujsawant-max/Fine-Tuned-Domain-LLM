"""Generate the base-vs-FinSage benchmark report (Phase 7).

Run with ``make report``. Reads result files written by the eval scripts (when
they exist) and renders ``reports/benchmark_report.md``. In Phase 1 it produces
an empty-but-valid report so the pipeline shape is testable end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

from finsage.config import get_settings
from finsage.evaluation.report_generator import render_report
from finsage.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

REPORTS_DIR = Path("reports")


def _load_results(path: Path) -> dict[str, float]:
    """Load a metric results JSON file if it exists.

    Args:
        path: Path to a JSON file mapping metric keys to float values.

    Returns:
        The loaded mapping, or an empty dict if the file is absent.
    """
    if not path.exists():
        logger.warning("Results file %s not found; using empty metrics.", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return {str(k): float(v) for k, v in json.load(fh).items()}


def main() -> None:
    """Render the benchmark report from available result files."""
    setup_logging(get_settings().log_level)
    base = _load_results(REPORTS_DIR / "base_results.json")
    fine = _load_results(REPORTS_DIR / "finetuned_results.json")

    report = render_report(base, fine)
    out_path = REPORTS_DIR / "benchmark_report.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote benchmark report to %s", out_path)


if __name__ == "__main__":
    main()
