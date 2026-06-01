"""QLoRA fine-tuning utilities for FinSage-7B (Phase 5).

All heavy dependencies (torch, transformers, peft, trl, datasets, bitsandbytes)
are imported **lazily inside functions**, so this module imports cleanly on a
plain CPU machine. The dry-run path in the CLI never calls the model-loading
functions, so it works without these dependencies installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from finsage.logging_utils import get_logger
from finsage.training.data_formatter import (
    format_dataset_for_sft,
    validate_training_example,
)

logger = get_logger(__name__)

_MISSING_DEPS_MSG = (
    "Real QLoRA training requires torch, transformers, peft, trl, datasets, and "
    "bitsandbytes. Install training dependencies with "
    "pip install -e '.[training,ml]'"
)


def _require(*module_names: str) -> None:
    """Ensure the named modules are importable, else raise a helpful error.

    Args:
        *module_names: Module names that must be importable.

    Raises:
        ImportError: If any module is missing, with install guidance.
    """
    import importlib.util

    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"{_MISSING_DEPS_MSG} (missing: {', '.join(missing)})")


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config file into a dict.

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
        return dict(yaml.safe_load(fh) or {})


def build_bnb_config(config: dict[str, Any]) -> Any:
    """Build a BitsAndBytes 4-bit quantization config.

    Args:
        config: The ``quantization`` config section (keys such as
            ``bnb_4bit_quant_type`` and ``bnb_4bit_compute_dtype``).

    Returns:
        A ``transformers.BitsAndBytesConfig`` for 4-bit NF4 loading.

    Raises:
        ImportError: If transformers/bitsandbytes are not installed.
    """
    _require("torch", "transformers", "bitsandbytes")
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = getattr(torch, str(config.get("bnb_4bit_compute_dtype", "bfloat16")))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(config: dict[str, Any]) -> Any:
    """Build a PEFT LoRA config.

    Args:
        config: The LoRA config (``r``, ``lora_alpha``, ``lora_dropout``,
            ``bias``, ``task_type``, ``target_modules``).

    Returns:
        A ``peft.LoraConfig``.

    Raises:
        ImportError: If peft is not installed.
    """
    _require("peft")
    from peft import LoraConfig

    return LoraConfig(
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=str(config.get("bias", "none")),
        task_type=str(config.get("task_type", "CAUSAL_LM")),
        target_modules=list(
            config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )


def load_model_and_tokenizer(
    model_id: str,
    training_config: dict[str, Any],
    lora_config: dict[str, Any],
    use_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load the base model + tokenizer and attach a LoRA adapter.

    Args:
        model_id: Hugging Face model id.
        training_config: The full training config dict (uses ``model`` and
            ``quantization`` sections).
        lora_config: The LoRA config dict.
        use_4bit: Whether to load the model in 4-bit and prepare for k-bit training.

    Returns:
        A ``(peft_model, tokenizer)`` tuple.

    Raises:
        ImportError: If torch/transformers/peft are not installed.
    """
    _require("torch", "transformers", "peft")
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = training_config.get("model", {})
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = getattr(torch, str(model_cfg.get("torch_dtype", "bfloat16")), torch.bfloat16)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if use_4bit:
        model_kwargs["quantization_config"] = build_bnb_config(
            training_config.get("quantization", {})
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_config.get("training", {}).get(
                "gradient_checkpointing", True
            ),
        )

    model = get_peft_model(model, build_lora_config(lora_config))
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    logger.info("Loaded model %s with LoRA adapter (4-bit=%s)", model_id, use_4bit)
    return model, tokenizer


def load_jsonl_dataset(train_file: Path | str, validation_file: Path | str) -> Any:
    """Load train/validation JSONL files into a Hugging Face ``DatasetDict``.

    Args:
        train_file: Path to the training JSONL.
        validation_file: Path to the validation JSONL.

    Returns:
        A ``datasets.DatasetDict`` with ``train`` and ``validation`` splits.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
    """
    _require("datasets")
    from datasets import load_dataset

    return load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(validation_file)},
    )


def build_training_args(config: dict[str, Any]) -> Any:
    """Build training arguments from the config.

    Prefers TRL's ``SFTConfig`` (a ``TrainingArguments`` subclass) so that
    SFT-specific fields (``max_seq_length``, ``packing``, ``dataset_text_field``)
    are carried in a version-correct way; falls back to plain
    ``transformers.TrainingArguments`` when TRL is unavailable. Only fields that
    the installed class actually accepts are passed, so this is robust across
    TRL/transformers versions.

    Args:
        config: The full training config dict (uses ``training`` + ``sft``).

    Returns:
        An ``SFTConfig`` (modern TRL) or ``TrainingArguments`` instance.

    Raises:
        ImportError: If transformers is not installed.
    """
    _require("transformers")
    import inspect

    training = dict(config.get("training", {}))
    sft = dict(config.get("sft", {}))

    try:
        from trl import SFTConfig

        params = inspect.signature(SFTConfig.__init__).parameters
        kwargs = dict(training)
        for field in ("max_seq_length", "packing", "dataset_text_field"):
            if field in params and field in sft:
                kwargs[field] = sft[field]
        return SFTConfig(**kwargs)
    except ImportError:
        from transformers import TrainingArguments

        return TrainingArguments(**training)


def build_sft_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    training_args: Any,
    config: dict[str, Any],
) -> Any:
    """Build a TRL ``SFTTrainer``, adapting to the installed TRL signature.

    Modern TRL (>=0.12) takes the tokenizer as ``processing_class`` and carries
    SFT fields on ``SFTConfig`` (built in :func:`build_training_args`); older TRL
    accepts ``tokenizer``/``dataset_text_field``/``max_seq_length``/``packing``
    directly. We introspect the constructor and pass only what it accepts.

    Args:
        model: The (PEFT) model to train.
        tokenizer: The tokenizer.
        dataset: A ``DatasetDict`` with ``train``/``validation`` and a ``text`` field.
        training_args: The training arguments (``SFTConfig`` or ``TrainingArguments``).
        config: The full training config (uses the ``sft`` section).

    Returns:
        A configured ``SFTTrainer``.

    Raises:
        ImportError: If trl is not installed.
    """
    _require("trl")
    import inspect

    from trl import SFTTrainer

    params = inspect.signature(SFTTrainer.__init__).parameters
    sft_cfg = config.get("sft", {})

    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("validation"),
    }
    # Tokenizer argument was renamed processing_class in modern TRL.
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    # Legacy TRL accepted these on the trainer; modern TRL takes them via SFTConfig.
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = str(sft_cfg.get("dataset_text_field", "text"))
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = int(sft_cfg.get("max_seq_length", 2048))
    if "packing" in params:
        kwargs["packing"] = bool(sft_cfg.get("packing", True))

    return SFTTrainer(**kwargs)


def _final_losses(trainer: Any) -> tuple[float | None, float | None]:
    """Extract the final train/eval loss from a trainer's log history.

    Args:
        trainer: A trained ``SFTTrainer``/``Trainer``.

    Returns:
        A ``(final_train_loss, final_eval_loss)`` tuple; entries are ``None``
        when unavailable.
    """
    train_loss: float | None = None
    eval_loss: float | None = None
    for record in getattr(getattr(trainer, "state", None), "log_history", []) or []:
        if "loss" in record:
            train_loss = float(record["loss"])
        if "eval_loss" in record:
            eval_loss = float(record["eval_loss"])
    return train_loss, eval_loss


def train(
    train_file: Path | str,
    validation_file: Path | str,
    model_id: str,
    output_dir: Path | str,
    training_config_path: Path | str,
    lora_config_path: Path | str,
    use_4bit: bool = True,
    resume_from_checkpoint: str | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    """Run the full QLoRA fine-tuning flow and save the adapter.

    Args:
        train_file: Path to the training JSONL.
        validation_file: Path to the validation JSONL.
        model_id: Base model id.
        output_dir: Directory to write the adapter, tokenizer, and summary.
        training_config_path: Path to the training YAML config.
        lora_config_path: Path to the LoRA YAML config.
        use_4bit: Whether to use 4-bit QLoRA loading.
        resume_from_checkpoint: Optional checkpoint path to resume from.
        max_train_samples: Optional cap on training examples.
        max_eval_samples: Optional cap on validation examples.

    Returns:
        The training summary dict (also written to ``training_summary.json``).

    Raises:
        ImportError: If training dependencies are missing.
        ValueError: If any dataset example fails schema validation.
    """
    training_config = load_yaml_config(training_config_path)
    lora_config = load_yaml_config(lora_config_path)

    dataset = load_jsonl_dataset(train_file, validation_file)

    for split in ("train", "validation"):
        for example in dataset[split]:
            errors = validate_training_example(example)
            if errors:
                raise ValueError(f"Invalid {split} example {example.get('id', '?')}: {errors}")

    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(max_train_samples, len(dataset["train"])))
        )
    if max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )

    dataset = format_dataset_for_sft(dataset)

    model, tokenizer = load_model_and_tokenizer(
        model_id, training_config, lora_config, use_4bit=use_4bit
    )

    training_config.setdefault("training", {})["output_dir"] = str(output_dir)
    training_args = build_training_args(training_config)
    trainer = build_sft_trainer(model, tokenizer, dataset, training_args, training_config)

    logger.info("Starting training (resume=%s)", resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    final_train_loss, final_eval_loss = _final_losses(trainer)
    summary = {
        "model_id": model_id,
        "output_dir": str(out_dir),
        "train_file": str(train_file),
        "validation_file": str(validation_file),
        "num_train_examples": len(dataset["train"]),
        "num_eval_examples": len(dataset["validation"]),
        "lora_r": lora_config.get("r"),
        "lora_alpha": lora_config.get("lora_alpha"),
        "learning_rate": training_config.get("training", {}).get("learning_rate"),
        "epochs": training_config.get("training", {}).get("num_train_epochs"),
        "max_seq_length": training_config.get("sft", {}).get("max_seq_length"),
        "packing": training_config.get("sft", {}).get("packing"),
        "report_to": training_config.get("training", {}).get("report_to"),
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Training complete; adapter saved to %s", out_dir)
    return summary
