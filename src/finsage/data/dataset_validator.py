"""Validate instruction-dataset JSONL files and their splits.

Checks per-example structure, JSONL parse success, duplicate ids across splits,
train/test company leakage, and task-type coverage. Dependency-free and CPU-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finsage.data.instruction_builder import TASK_TYPES
from finsage.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = ("id", "instruction", "input", "output", "task_type", "metadata")
VALID_SPLITS = {"train", "validation", "test"}


class DatasetValidator:
    """Validates instruction-dataset examples, files, and splits."""

    def validate_example(self, example: dict[str, Any]) -> list[str]:
        """Validate a single example.

        Args:
            example: The parsed example dict.

        Returns:
            A list of error strings; empty if the example is valid.
        """
        errors: list[str] = []
        for field in REQUIRED_FIELDS:
            if field not in example:
                errors.append(f"missing field '{field}'")
            elif example[field] is None:
                errors.append(f"null field '{field}'")

        if not str(example.get("id", "")).strip():
            errors.append("empty id")
        if not str(example.get("instruction", "")).strip():
            errors.append("empty instruction")
        if len(str(example.get("input", ""))) == 0:
            errors.append("empty input")
        if len(str(example.get("output", ""))) == 0:
            errors.append("empty output")

        task_type = example.get("task_type")
        if task_type is not None and task_type not in TASK_TYPES:
            errors.append(f"invalid task_type '{task_type}'")

        if "metadata" in example and not isinstance(example["metadata"], dict):
            errors.append("metadata is not an object")

        split = example.get("split")
        if split is not None and split not in VALID_SPLITS:
            errors.append(f"invalid split '{split}'")

        return errors

    def validate_file(self, path: Path | str) -> dict[str, Any]:
        """Validate every example in a JSONL file.

        Args:
            path: Path to the JSONL file.

        Returns:
            A report dict with ``path``, ``total_examples``, ``valid_examples``,
            ``invalid_examples``, and ``errors`` (list of ``{line, errors}``).
        """
        file_path = Path(path)
        report: dict[str, Any] = {
            "path": str(file_path),
            "total_examples": 0,
            "valid_examples": 0,
            "invalid_examples": 0,
            "errors": [],
        }
        if not file_path.exists():
            report["errors"].append({"line": 0, "errors": ["file not found"]})
            return report

        with file_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                report["total_examples"] += 1
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as exc:
                    report["invalid_examples"] += 1
                    report["errors"].append(
                        {"line": line_no, "errors": [f"JSON parse error: {exc}"]}
                    )
                    continue
                errors = self.validate_example(example)
                if errors:
                    report["invalid_examples"] += 1
                    report["errors"].append({"line": line_no, "errors": errors})
                else:
                    report["valid_examples"] += 1
        return report

    @staticmethod
    def _load_ids_and_tickers(path: Path) -> tuple[list[str], set[str], set[str]]:
        """Load ids, the id set, and company tickers from a JSONL file.

        Args:
            path: Path to the JSONL file.

        Returns:
            A tuple ``(ids, id_set, tickers)``.
        """
        ids: list[str] = []
        tickers: set[str] = set()
        if not path.exists():
            return ids, set(), tickers
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ids.append(str(example.get("id", "")))
                meta = example.get("metadata", {})
                tickers.add(str(meta.get("ticker") or meta.get("cik") or "UNK").upper())
        return ids, set(ids), tickers

    def validate_splits(
        self,
        train_path: Path | str,
        validation_path: Path | str,
        test_path: Path | str,
    ) -> dict[str, Any]:
        """Validate all three split files and cross-split invariants.

        Args:
            train_path: Path to ``train.jsonl``.
            validation_path: Path to ``validation.jsonl``.
            test_path: Path to ``test.jsonl``.

        Returns:
            A report dict with per-file reports, duplicate-id detection,
            train/test ticker overlap, task-type coverage, and an overall
            ``passed`` flag.
        """
        files = {
            "train": Path(train_path),
            "validation": Path(validation_path),
            "test": Path(test_path),
        }
        file_reports = {name: self.validate_file(p) for name, p in files.items()}

        ids_train, set_train, tick_train = self._load_ids_and_tickers(files["train"])
        ids_val, set_val, tick_val = self._load_ids_and_tickers(files["validation"])
        ids_test, set_test, tick_test = self._load_ids_and_tickers(files["test"])

        all_ids = ids_train + ids_val + ids_test
        duplicate_ids = sorted({i for i in all_ids if all_ids.count(i) > 1})

        train_test_overlap = sorted(tick_train & tick_test)

        present_task_types: set[str] = set()
        for path in files.values():
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            present_task_types.add(json.loads(line).get("task_type", ""))
                        except json.JSONDecodeError:
                            continue
        missing_task_types = sorted(set(TASK_TYPES) - present_task_types)

        files_valid = all(r["invalid_examples"] == 0 for r in file_reports.values())
        passed = files_valid and not duplicate_ids and not train_test_overlap

        report = {
            "files": file_reports,
            "duplicate_ids": duplicate_ids,
            "train_test_ticker_overlap": train_test_overlap,
            "task_types_present": sorted(present_task_types - {""}),
            "task_types_missing": missing_task_types,
            "all_task_types_present": not missing_task_types,
            "passed": passed,
        }
        logger.info("Split validation passed=%s", passed)
        return report

    @staticmethod
    def write_validation_report(report: dict[str, Any], path: Path | str) -> Path:
        """Write a validation report as JSON.

        Args:
            report: The report dict.
            path: Destination JSON path.

        Returns:
            The path written to.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote validation report to %s", out_path)
        return out_path
