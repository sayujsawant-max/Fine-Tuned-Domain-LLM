"""Tests for the training CLI dry-run (no model loading)."""

from __future__ import annotations

from pathlib import Path

from training.train import app
from typer.testing import CliRunner

runner = CliRunner()

CONFIG = "configs/training_config.yaml"
LORA_CONFIG = "configs/lora_config.yaml"


def _args(train: Path | str, val: Path | str, out: Path) -> list[str]:
    """Build the dry-run CLI argument list."""
    return [
        "--train-file",
        str(train),
        "--validation-file",
        str(val),
        "--output-dir",
        str(out),
        "--config",
        CONFIG,
        "--lora-config",
        LORA_CONFIG,
        "--dry-run",
    ]


def test_dry_run_succeeds(train_sample_file, validation_sample_file, tmp_path):
    """Dry-run validates fixtures and previews formatting (exit 0)."""
    result = runner.invoke(app, _args(train_sample_file, validation_sample_file, tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert "Dry-run OK" in result.output


def test_missing_train_file(validation_sample_file, tmp_path):
    """A missing train file produces a clean error and non-zero exit."""
    result = runner.invoke(
        app, _args(tmp_path / "nope.jsonl", validation_sample_file, tmp_path / "o")
    )
    assert result.exit_code == 1
    assert "Training file not found" in result.output


def test_missing_validation_file(train_sample_file, tmp_path):
    """A missing validation file produces a clean error and non-zero exit."""
    result = runner.invoke(app, _args(train_sample_file, tmp_path / "nope.jsonl", tmp_path / "o"))
    assert result.exit_code == 1
    assert "Validation file not found" in result.output


def test_invalid_jsonl(validation_sample_file, tmp_path):
    """Malformed JSONL produces a clean error and non-zero exit."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"instruction": "x"}\n{not valid json}\n', encoding="utf-8")
    result = runner.invoke(app, _args(bad, validation_sample_file, tmp_path / "o"))
    assert result.exit_code == 1
    assert "invalid JSON" in result.output
