"""QLoRA trainer orchestration (Phase 6 stub).

This module defines the training contract and config loading. The actual fit
loop depends on the ``training`` optional group (torch, peft, trl, bitsandbytes)
and is implemented in Phase 6. Heavy imports are deferred into method bodies so
this file imports cleanly on a CPU-only machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class QLoRATrainer:
    """Coordinates QLoRA fine-tuning of the base model.

    Args:
        lora_config_path: Path to the LoRA YAML config.
        training_config_path: Path to the training YAML config.
    """

    lora_config_path: str | Path = "configs/lora_config.yaml"
    training_config_path: str | Path = "configs/training_config.yaml"

    def load_configs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load and return the LoRA and training configs.

        Returns:
            A ``(lora_config, training_config)`` tuple of plain dicts.

        Raises:
            FileNotFoundError: If either config file is missing.
        """
        lora = self._read_yaml(self.lora_config_path)
        training = self._read_yaml(self.training_config_path)
        logger.info("Loaded LoRA and training configs")
        return lora, training

    @staticmethod
    def _read_yaml(path: str | Path) -> dict[str, Any]:
        """Read a YAML file into a dict.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed mapping (empty dict for an empty file).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data)

    def train(self) -> None:
        """Run the QLoRA fine-tuning loop.

        Raises:
            NotImplementedError: Always, until Phase 6. Requires the ``training``
                optional dependency group.
        """
        raise NotImplementedError(
            "QLoRA training lands in Phase 6 (install the 'training' extras)."
        )
