"""Entry point for QLoRA fine-tuning (Phase 6).

Run with ``make train`` or ``python training/train.py``. Requires the
``training`` optional dependency group (torch, peft, trl, bitsandbytes).
"""

from __future__ import annotations

from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging
from finsage.training.qlora_trainer import QLoRATrainer

logger = get_logger(__name__)


def main() -> None:
    """Load configs and launch the QLoRA training run."""
    setup_logging(get_settings().log_level)
    trainer = QLoRATrainer()
    lora_config, training_config = trainer.load_configs()
    logger.info(
        "Loaded configs (lora keys=%s, training keys=%s)",
        list(lora_config),
        list(training_config),
    )
    trainer.train()


if __name__ == "__main__":
    main()
