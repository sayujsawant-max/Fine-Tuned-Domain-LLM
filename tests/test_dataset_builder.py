"""Tests for DatasetBuilder."""

from __future__ import annotations

import json
from pathlib import Path

from finsage.data.dataset_builder import DatasetBuilder
from finsage.data.instruction_builder import TASK_TYPES


def _builder(processed_manifest_path: Path, tmp_path: Path) -> DatasetBuilder:
    """Build a DatasetBuilder pointed at the fixture manifest."""
    return DatasetBuilder(
        processed_manifest_path=processed_manifest_path,
        output_dir=tmp_path / "datasets",
        random_seed=42,
    )


def test_loads_processed_manifest(processed_manifest_path, tmp_path):
    """The processed manifest loads all fixture rows."""
    rows = _builder(processed_manifest_path, tmp_path).load_processed_manifest()
    assert len(rows) == 7
    assert {r["ticker"] for r in rows} == {"AAPL", "MSFT", "NVDA", "TSLA"}


def test_reads_processed_text(processed_manifest_path, tmp_path):
    """Processed section text is read from disk."""
    builder = _builder(processed_manifest_path, tmp_path)
    rows = builder.load_processed_manifest()
    text = builder.read_processed_text(rows[0])
    assert "ACME Devices" in text


def test_applicable_task_types_by_section():
    """Section -> task-type mapping matches the spec."""
    assert "risk_summary" in DatasetBuilder.applicable_task_types("risk_factors")
    assert "mda_explanation" in DatasetBuilder.applicable_task_types("mda")
    assert "metric_extraction" in DatasetBuilder.applicable_task_types("financial_statements")
    assert "hallucination_detection" in DatasetBuilder.applicable_task_types("market_risk")
    assert DatasetBuilder.applicable_task_types("unknown") == []
    # Business does not include risk_summary.
    assert "risk_summary" not in DatasetBuilder.applicable_task_types("business")


def test_builds_examples_from_fixture(processed_manifest_path, tmp_path):
    """Examples are built across sections and all 10 task types appear."""
    builder = _builder(processed_manifest_path, tmp_path)
    examples = builder.build_all_examples()
    assert len(examples) > 0
    # The fixture's sections (business, risk_factors, mda, financial_statements)
    # together cover all 10 task types.
    assert {e["task_type"] for e in examples} == set(TASK_TYPES)


def test_split_prevents_train_test_company_overlap(processed_manifest_path, tmp_path):
    """company_holdout produces disjoint train/test companies."""
    builder = _builder(processed_manifest_path, tmp_path)
    examples = builder.build_all_examples()
    splits = builder.split_examples(examples, strategy="company_holdout")

    def companies(name: str) -> set[str]:
        return {e["metadata"]["ticker"] for e in splits[name]}

    assert companies("train") & companies("test") == set()
    assert all(e["split"] in {"train", "validation", "test"} for e in examples)
    leakage = builder.check_leakage(splits)
    assert leakage["passed"] is True
    assert leakage["train_test_ticker_overlap"] == []


def test_write_jsonl_is_valid(processed_manifest_path, tmp_path):
    """write_jsonl produces one valid JSON object per line."""
    builder = _builder(processed_manifest_path, tmp_path)
    examples = builder.build_all_examples(max_examples=5)
    path = builder.write_jsonl(examples, tmp_path / "out.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(examples)
    assert all(json.loads(line)["id"] for line in lines)


def test_write_dataset_stats_contains_required_fields(processed_manifest_path, tmp_path):
    """Stats JSON includes counts, per-split detail, and the leakage check."""
    builder = _builder(processed_manifest_path, tmp_path)
    examples = builder.build_all_examples()
    splits = builder.split_examples(examples)
    stats_path = builder.write_dataset_stats(splits, tmp_path / "stats.json")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    for key in (
        "total_examples",
        "examples_per_split",
        "examples_per_task_type",
        "examples_per_section",
        "average_input_length",
        "average_output_length",
        "max_input_length",
        "min_input_length",
        "per_split",
        "leakage_check",
    ):
        assert key in stats
    assert stats["leakage_check"]["passed"] is True
    assert set(stats["per_split"]) == {"train", "validation", "test"}


def test_build_end_to_end_writes_all_outputs(processed_manifest_path, tmp_path):
    """The full build writes train/val/test, stats, and manifest with no leakage."""
    builder = _builder(processed_manifest_path, tmp_path)
    summary = builder.build(max_examples=200, strategy="company_holdout")
    out = tmp_path / "datasets"
    assert (out / "train.jsonl").exists()
    assert (out / "validation.jsonl").exists()
    assert (out / "test.jsonl").exists()
    assert (out / "dataset_stats.json").exists()
    assert (out / "dataset_manifest.jsonl").exists()
    assert summary["leakage_check"]["passed"] is True
