"""Tests for DatasetValidator."""

from __future__ import annotations

import json
from pathlib import Path

from finsage.data.dataset_validator import DatasetValidator

validator = DatasetValidator()


def _valid_example(
    example_id: str = "AAPL-2022-10-K-x-mda-0-filing_qa", ticker: str = "AAPL"
) -> dict:
    """Return a structurally valid example."""
    return {
        "id": example_id,
        "instruction": "Answer using the excerpt.",
        "input": "Revenue grew 8% this year.",
        "output": "Revenue grew 8%.",
        "task_type": "filing_qa",
        "source": "AAPL 2022 10-K mda",
        "metadata": {"ticker": ticker, "section": "mda"},
    }


def _write_jsonl(path: Path, examples: list[dict]) -> Path:
    """Write examples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in examples) + "\n", encoding="utf-8")
    return path


def test_validates_correct_example():
    """A well-formed example yields no errors."""
    assert validator.validate_example(_valid_example()) == []


def test_catches_missing_and_empty_fields():
    """Missing and empty required fields are reported."""
    bad = _valid_example()
    del bad["output"]
    errors = validator.validate_example(bad)
    assert any("output" in e for e in errors)

    empty = _valid_example()
    empty["input"] = ""
    assert any("empty input" in e for e in validator.validate_example(empty))


def test_catches_invalid_task_type_and_split():
    """Invalid task type and split values are reported."""
    bad = _valid_example()
    bad["task_type"] = "nope"
    bad["split"] = "holdout"
    errors = validator.validate_example(bad)
    assert any("invalid task_type" in e for e in errors)
    assert any("invalid split" in e for e in errors)


def test_validate_file(tmp_path):
    """File validation counts valid and invalid examples."""
    path = _write_jsonl(tmp_path / "train.jsonl", [_valid_example("a"), _valid_example("b")])
    report = validator.validate_file(path)
    assert report["total_examples"] == 2
    assert report["valid_examples"] == 2
    assert report["invalid_examples"] == 0


def test_detects_duplicate_ids_across_splits(tmp_path):
    """Duplicate ids spanning splits are detected."""
    train = _write_jsonl(tmp_path / "train.jsonl", [_valid_example("dup", "AAPL")])
    val = _write_jsonl(tmp_path / "val.jsonl", [_valid_example("v1", "AAPL")])
    test = _write_jsonl(tmp_path / "test.jsonl", [_valid_example("dup", "MSFT")])
    report = validator.validate_splits(train, val, test)
    assert "dup" in report["duplicate_ids"]
    assert report["passed"] is False


def test_detects_train_test_ticker_leakage(tmp_path):
    """Train/test company overlap is detected and fails validation."""
    train = _write_jsonl(tmp_path / "train.jsonl", [_valid_example("t1", "AAPL")])
    val = _write_jsonl(tmp_path / "val.jsonl", [_valid_example("v1", "MSFT")])
    test = _write_jsonl(tmp_path / "test.jsonl", [_valid_example("e1", "AAPL")])
    report = validator.validate_splits(train, val, test)
    assert "AAPL" in report["train_test_ticker_overlap"]
    assert report["passed"] is False


def test_clean_splits_pass(tmp_path):
    """Disjoint, well-formed splits pass validation."""
    train = _write_jsonl(tmp_path / "train.jsonl", [_valid_example("t1", "AAPL")])
    val = _write_jsonl(tmp_path / "val.jsonl", [_valid_example("v1", "MSFT")])
    test = _write_jsonl(tmp_path / "test.jsonl", [_valid_example("e1", "NVDA")])
    report = validator.validate_splits(train, val, test)
    assert not report["duplicate_ids"]
    assert not report["train_test_ticker_overlap"]
    assert report["passed"] is True


def test_writes_validation_report(tmp_path):
    """The report is written as JSON."""
    train = _write_jsonl(tmp_path / "train.jsonl", [_valid_example("t1", "AAPL")])
    val = _write_jsonl(tmp_path / "val.jsonl", [_valid_example("v1", "MSFT")])
    test = _write_jsonl(tmp_path / "test.jsonl", [_valid_example("e1", "NVDA")])
    report = validator.validate_splits(train, val, test)
    out = validator.write_validation_report(report, tmp_path / "report.json")
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True
