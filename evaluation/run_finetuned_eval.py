"""Evaluate the fine-tuned FinSage-7B model on the test set (Phase 7).

Reuses the Phase 4 :class:`EvalRunner`, but with a generator that loads the base
model plus the trained LoRA adapter. The adapter-loading path lands in Phase 7;
until then this entry point explains how to run it once an adapter exists.
"""

from __future__ import annotations

from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Run the fine-tuned (FinSage-7B) evaluation pass.

    Raises:
        NotImplementedError: Until Phase 7. Fine-tuned evaluation reuses the
            Phase 4 ``EvalRunner`` with a transformers generator pointed at the
            base model + trained LoRA adapter (``ADAPTER_PATH``).
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Phase 7 stub: fine-tuned eval will reuse EvalRunner with the adapter at %s",
        settings.adapter_path,
    )
    raise NotImplementedError(
        "Fine-tuned evaluation lands in Phase 7 (requires a trained adapter and "
        "the 'ml'/'training' extras)."
    )


if __name__ == "__main__":
    main()
