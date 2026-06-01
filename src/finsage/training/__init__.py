"""QLoRA fine-tuning components (Phase 5).

Heavy GPU libraries (torch, transformers, peft, trl, bitsandbytes, datasets) are
imported lazily inside functions, so importing this package on a CPU-only
machine stays cheap. The pure-Python data formatter and config loaders are safe
to import directly.
"""

from __future__ import annotations

from finsage.training.callbacks import build_default_callbacks
from finsage.training.data_formatter import (
    count_tokens_approx,
    format_dataset_for_sft,
    format_sft_example,
    validate_training_example,
)
from finsage.training.qlora_trainer import (
    build_bnb_config,
    build_lora_config,
    build_sft_trainer,
    build_training_args,
    load_jsonl_dataset,
    load_model_and_tokenizer,
    load_yaml_config,
    train,
)

__all__ = [
    "build_bnb_config",
    "build_default_callbacks",
    "build_lora_config",
    "build_sft_trainer",
    "build_training_args",
    "count_tokens_approx",
    "format_dataset_for_sft",
    "format_sft_example",
    "load_jsonl_dataset",
    "load_model_and_tokenizer",
    "load_yaml_config",
    "train",
    "validate_training_example",
]
