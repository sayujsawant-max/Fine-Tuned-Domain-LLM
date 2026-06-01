"""Tests for QLoRA trainer utilities (no model downloads, no GPU)."""

from __future__ import annotations

import importlib.util

import pytest

from finsage.training.qlora_trainer import (
    build_bnb_config,
    build_lora_config,
    load_jsonl_dataset,
    load_yaml_config,
)

_HAS_PEFT = importlib.util.find_spec("peft") is not None
_HAS_BNB = (
    importlib.util.find_spec("bitsandbytes") is not None
    and importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)
_HAS_DATASETS = importlib.util.find_spec("datasets") is not None

LORA_CFG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def test_load_yaml_config(tmp_path):
    """YAML config loads into a dict."""
    p = tmp_path / "cfg.yaml"
    p.write_text("a: 1\nb:\n  - x\n  - y\n", encoding="utf-8")
    cfg = load_yaml_config(p)
    assert cfg["a"] == 1 and cfg["b"] == ["x", "y"]


def test_load_yaml_config_missing(tmp_path):
    """A missing config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "nope.yaml")


def test_build_lora_config():
    """build_lora_config returns a LoRA config, or raises a helpful error."""
    if _HAS_PEFT:
        cfg = build_lora_config(LORA_CFG)
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert "q_proj" in cfg.target_modules
    else:
        with pytest.raises(ImportError, match=r"training,ml"):
            build_lora_config(LORA_CFG)


def test_build_bnb_config():
    """build_bnb_config returns a 4-bit config, or raises a helpful error."""
    quant = {
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    if _HAS_BNB:
        cfg = build_bnb_config(quant)
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
    else:
        with pytest.raises(ImportError, match=r"training,ml"):
            build_bnb_config(quant)


def test_load_jsonl_dataset(train_sample_file, validation_sample_file):
    """load_jsonl_dataset returns a DatasetDict, or raises a helpful error."""
    if _HAS_DATASETS:
        ds = load_jsonl_dataset(train_sample_file, validation_sample_file)
        assert len(ds["train"]) == 10
        assert len(ds["validation"]) == 5
    else:
        with pytest.raises(ImportError, match=r"training,ml"):
            load_jsonl_dataset(train_sample_file, validation_sample_file)
