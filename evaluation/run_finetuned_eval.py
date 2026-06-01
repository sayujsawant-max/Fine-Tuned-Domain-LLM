"""Evaluate the fine-tuned FinSage-7B model on the test set (Phase 7).

Run with ``make eval-finetuned``. Requires the ``ml`` (and ``training`` for the
adapter) optional dependency groups.
"""

from __future__ import annotations

from finsage.config import get_settings
from finsage.evaluation.runner import EvalRunner
from finsage.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Run the fine-tuned (FinSage-7B) evaluation pass."""
    setup_logging(get_settings().log_level)
    runner = EvalRunner(model_name="finsage-7b")
    config = runner.load_config()
    logger.info("Loaded eval config with tasks: %s", list(config.get("tasks", {})))
    results = runner.run()
    logger.info("FinSage-7B results: %s", results)


if __name__ == "__main__":
    main()
