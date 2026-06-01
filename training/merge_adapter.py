"""Merge a trained LoRA adapter into the base model (Phase 6/8).

Produces a standalone merged checkpoint suitable for vLLM serving. Requires the
``training`` optional dependency group.
"""

from __future__ import annotations

from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def merge_adapter(base_model_id: str, adapter_path: str, output_path: str) -> None:
    """Merge a LoRA adapter into the base model and save the result.

    Args:
        base_model_id: Hugging Face id of the base model.
        adapter_path: Path to the trained LoRA adapter.
        output_path: Directory to write the merged model into.

    Raises:
        NotImplementedError: Always, until Phase 6. Requires the ``training``
            extras (peft + transformers).
    """
    raise NotImplementedError("Adapter merging lands in Phase 6 (install the 'training' extras).")


def main() -> None:
    """Merge the configured adapter into the base model."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Phase 6 stub: would merge adapter %s into %s -> %s",
        settings.adapter_path,
        settings.model_id,
        settings.merged_model_path,
    )
    merge_adapter(
        base_model_id=settings.model_id,
        adapter_path=settings.adapter_path,
        output_path=settings.merged_model_path,
    )


if __name__ == "__main__":
    main()
