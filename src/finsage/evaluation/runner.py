"""Evaluation orchestration (Phase 5 & 7).

:class:`EvalRunner` ties together a model under test, a test dataset, and the
metric functions in :mod:`finsage.evaluation.metrics`. Phase 1 ships the config
loading and result-shaping scaffold; model generation is wired up in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvalRunner:
    """Runs an evaluation pass over a test set for one model.

    Args:
        config_path: Path to the evaluation YAML config.
        model_name: Logical name of the model under test (e.g. ``"base-mistral-7b"``).
    """

    config_path: str | Path = "configs/eval_config.yaml"
    model_name: str = "base-mistral-7b"
    _config: dict[str, Any] = field(init=False, default_factory=dict)

    def load_config(self) -> dict[str, Any]:
        """Load the evaluation config from disk.

        Returns:
            The parsed config mapping.

        Raises:
            FileNotFoundError: If the config file is missing.
        """
        p = Path(self.config_path)
        if not p.exists():
            raise FileNotFoundError(f"Eval config not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            self._config = dict(yaml.safe_load(fh) or {})
        return self._config

    def run(self) -> dict[str, Any]:
        """Run the evaluation and return aggregated metrics.

        Returns:
            A results mapping keyed by task type.

        Raises:
            NotImplementedError: Always, until Phase 5. Model generation requires
                the ``ml`` (and, for the fine-tuned model, ``training``) extras.
        """
        raise NotImplementedError("EvalRunner.run lands in Phase 5 (install the 'ml' extras).")
