"""Evaluate the base Mistral-7B model on the test set (Phase 5).

Run with ``make eval-baseline``. Requires the ``ml`` optional dependency group.
"""

from __future__ import annotations

from finsage.config import get_settings
from finsage.evaluation.runner import EvalRunner
from finsage.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Run the baseline (base model) evaluation pass."""
    setup_logging(get_settings().log_level)
    runner = EvalRunner(model_name="base-mistral-7b")
    config = runner.load_config()
    logger.info("Loaded eval config with tasks: %s", list(config.get("tasks", {})))
    results = runner.run()
    logger.info("Baseline results: %s", results)


if __name__ == "__main__":
    main()
