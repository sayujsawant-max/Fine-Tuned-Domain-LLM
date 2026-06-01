"""QLoRA fine-tuning components (Phase 6).

These modules import heavy GPU libraries (torch, peft, trl, bitsandbytes) only
inside functions, so importing the package on a CPU-only machine stays cheap.
"""

from __future__ import annotations

__all__ = ["QLoRATrainer", "InstructionDataCollator", "build_default_callbacks"]


def __getattr__(name: str) -> object:
    """Lazily expose trainer symbols without importing GPU deps at import time."""
    if name == "QLoRATrainer":
        from finsage.training.qlora_trainer import QLoRATrainer

        return QLoRATrainer
    if name == "InstructionDataCollator":
        from finsage.training.data_collator import InstructionDataCollator

        return InstructionDataCollator
    if name == "build_default_callbacks":
        from finsage.training.callbacks import build_default_callbacks

        return build_default_callbacks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
