"""Build leakage-safe instruction-tuning datasets from processed filings.

Reads the Phase 2 processed manifest, chunks each section, generates
template/extractive instruction examples across the applicable task types, and
splits them into train/validation/test using company-level (or time-based)
holdout so no company appears in both train and test. Deterministic given a seed.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from finsage.data.chunker import FilingChunker
from finsage.data.instruction_builder import TASK_TYPES, InstructionBuilder
from finsage.logging_utils import get_logger

logger = get_logger(__name__)

#: Which task types apply to each filing section.
SECTION_TASK_TYPES: dict[str, list[str]] = {
    "business": [
        "business_risk_identification",
        "revenue_driver_explanation",
        "filing_qa",
        "analyst_summary",
        "outlook_classification",
    ],
    "risk_factors": [
        "risk_summary",
        "business_risk_identification",
        "filing_qa",
        "analyst_summary",
        "outlook_classification",
        "hallucination_detection",
    ],
    "mda": [
        "mda_explanation",
        "metric_extraction",
        "yoy_comparison",
        "revenue_driver_explanation",
        "filing_qa",
        "analyst_summary",
        "outlook_classification",
    ],
    "market_risk": [
        "metric_extraction",
        "business_risk_identification",
        "filing_qa",
        "analyst_summary",
        "hallucination_detection",
    ],
    "financial_statements": [
        "metric_extraction",
        "yoy_comparison",
        "filing_qa",
        "analyst_summary",
    ],
}

SPLIT_NAMES = ("train", "validation", "test")


class DatasetBuilder:
    """Builds and splits the FinSage instruction dataset.

    Args:
        processed_manifest_path: Path to the Phase 2 processed manifest (JSONL).
        output_dir: Directory for dataset outputs.
        chunker: Chunker to use; a default :class:`FilingChunker` is created
            when ``None``.
        instruction_builder: Builder to use; a default
            :class:`InstructionBuilder` is created when ``None``.
        random_seed: Seed for deterministic splitting.
    """

    def __init__(
        self,
        processed_manifest_path: Path | str,
        output_dir: Path | str = "data/datasets",
        chunker: FilingChunker | None = None,
        instruction_builder: InstructionBuilder | None = None,
        random_seed: int = 42,
    ) -> None:
        self.processed_manifest_path = Path(processed_manifest_path)
        self.output_dir = Path(output_dir)
        self.chunker = chunker or FilingChunker()
        self.instruction_builder = instruction_builder or InstructionBuilder()
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_processed_manifest(self) -> list[dict[str, Any]]:
        """Load the processed manifest rows.

        Returns:
            The parsed manifest rows.

        Raises:
            FileNotFoundError: If the manifest does not exist.
        """
        if not self.processed_manifest_path.exists():
            raise FileNotFoundError(f"Processed manifest not found: {self.processed_manifest_path}")
        rows: list[dict[str, Any]] = []
        with self.processed_manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        logger.info("Loaded %d processed manifest row(s)", len(rows))
        return rows

    @staticmethod
    def read_processed_text(row: dict[str, Any]) -> str:
        """Read the processed section text referenced by a manifest row.

        Args:
            row: A processed manifest row containing ``processed_path``.

        Returns:
            The section text, or an empty string if the file is missing.
        """
        path = row.get("processed_path")
        if not path or not Path(path).exists():
            logger.warning("Missing processed file: %s", path)
            return ""
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def applicable_task_types(section: str) -> list[str]:
        """Return the task types applicable to a section.

        Args:
            section: The section name (e.g. ``"risk_factors"``).

        Returns:
            The applicable task types, or an empty list for unknown sections.
        """
        return list(SECTION_TASK_TYPES.get(section, []))

    # ------------------------------------------------------------------
    # Example generation
    # ------------------------------------------------------------------
    def build_all_examples(
        self,
        max_examples: int | None = None,
        task_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build instruction examples for every processed section.

        Args:
            max_examples: Optional cap on the total number of examples.
            task_types: Optional whitelist of task types to keep.

        Returns:
            The generated examples.
        """
        allow = set(task_types) if task_types else None
        examples: list[dict[str, Any]] = []

        for row in self.load_processed_manifest():
            section = str(row.get("section", ""))
            text = self.read_processed_text(row)
            if not text.strip():
                continue

            filing_date = str(row.get("filing_date", ""))
            metadata = {
                "ticker": row.get("ticker", "") or "",
                "cik": row.get("cik", "") or "",
                "form": row.get("form", "") or "",
                "section": section,
                "filing_date": filing_date,
                "report_date": row.get("report_date", "") or "",
                "year": filing_date[:4] if filing_date[:4].isdigit() else "",
                "accession_number": row.get("accession_number", "") or "",
                "accession_number_no_dashes": str(row.get("accession_number", "")).replace("-", ""),
                "source_url": row.get("source_url", "") or "",
            }

            section_tasks = self.applicable_task_types(section)
            if allow is not None:
                section_tasks = [t for t in section_tasks if t in allow]
            if not section_tasks:
                continue

            for chunk in self.chunker.chunk_text(text, metadata=metadata):
                for example in self.instruction_builder.build_examples_for_chunk(
                    chunk, task_types=section_tasks
                ):
                    examples.append(example)
                    if max_examples is not None and len(examples) >= max_examples:
                        logger.info("Reached max_examples=%d", max_examples)
                        return examples

        logger.info("Built %d example(s)", len(examples))
        return examples

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _company_key(example: dict[str, Any]) -> str:
        """Return the company key (ticker preferred, else CIK) for an example."""
        meta = example.get("metadata", {})
        return str(meta.get("ticker") or meta.get("cik") or "UNK").upper()

    @staticmethod
    def _year_key(example: dict[str, Any]) -> str:
        """Return the filing year for an example (or empty string)."""
        return str(example.get("metadata", {}).get("year", ""))

    def split_examples(
        self,
        examples: list[dict[str, Any]],
        train_ratio: float = 0.85,
        validation_ratio: float = 0.10,
        test_ratio: float = 0.05,
        strategy: str = "company_holdout",
    ) -> dict[str, list[dict[str, Any]]]:
        """Split examples into train/validation/test without leakage.

        Args:
            examples: The examples to split.
            train_ratio: Target train fraction.
            validation_ratio: Target validation fraction.
            test_ratio: Target test fraction.
            strategy: ``"company_holdout"`` (whole companies per split) or
                ``"time_holdout"`` (latest filing years to test).

        Returns:
            A mapping ``{"train": [...], "validation": [...], "test": [...]}``
            with a ``split`` field set on every example.

        Raises:
            ValueError: If ``strategy`` is unknown.
        """
        if strategy == "company_holdout":
            groups = self._group_by(examples, self._company_key)
        elif strategy == "time_holdout":
            groups = self._group_by(examples, self._year_key)
        else:
            raise ValueError(f"Unknown split strategy: {strategy!r}")

        keys = sorted(groups)
        if strategy == "time_holdout":
            # Latest years to test, earliest to train; reverse so test gets newest.
            ordered = sorted(keys, reverse=True)
            test_keys, val_keys, train_keys = self._allocate_keys(
                ordered, train_ratio, validation_ratio, test_ratio, newest_first=True
            )
        else:
            rng = random.Random(self.random_seed)
            shuffled = keys[:]
            rng.shuffle(shuffled)
            test_keys, val_keys, train_keys = self._allocate_keys(
                shuffled, train_ratio, validation_ratio, test_ratio
            )

        assignment = dict.fromkeys(train_keys, "train")
        assignment.update(dict.fromkeys(val_keys, "validation"))
        assignment.update(dict.fromkeys(test_keys, "test"))

        splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
        for key, group in groups.items():
            split_name = assignment.get(key, "train")
            for example in group:
                example["split"] = split_name
                splits[split_name].append(example)

        logger.info(
            "Split: train=%d validation=%d test=%d (strategy=%s)",
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
            strategy,
        )
        return splits

    @staticmethod
    def _group_by(examples: list[dict[str, Any]], key_fn: Any) -> dict[str, list[dict[str, Any]]]:
        """Group examples by a key function."""
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in examples:
            groups[key_fn(example)].append(example)
        return dict(groups)

    @staticmethod
    def _allocate_keys(
        keys: list[str],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        newest_first: bool = False,
    ) -> tuple[list[str], list[str], list[str]]:
        """Allocate group keys to (test, validation, train).

        Guarantees train keeps at least one group, and (when there are enough
        groups) validation and test each get at least one.

        Args:
            keys: Ordered group keys.
            train_ratio: Target train fraction.
            validation_ratio: Target validation fraction.
            test_ratio: Target test fraction.
            newest_first: If ``True``, ``keys`` are ordered newest-first so the
                leading keys become the test set (time holdout).

        Returns:
            A ``(test_keys, validation_keys, train_keys)`` tuple.
        """
        n = len(keys)
        if n <= 1:
            return [], [], keys[:]

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * validation_ratio))
        # Always leave at least one group for train.
        while n_test + n_val > n - 1:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        test_keys = keys[:n_test]
        val_keys = keys[n_test : n_test + n_val]
        train_keys = keys[n_test + n_val :]
        return test_keys, val_keys, train_keys

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    @staticmethod
    def write_jsonl(examples: list[dict[str, Any]], path: Path | str) -> Path:
        """Write examples to a UTF-8 JSONL file.

        Args:
            examples: The examples to write.
            path: Destination path.

        Returns:
            The path written to.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for example in examples:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info("Wrote %d example(s) to %s", len(examples), out_path)
        return out_path

    def write_dataset(
        self,
        splits: dict[str, list[dict[str, Any]]],
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """Write train/validation/test JSONL files.

        Args:
            splits: The split mapping.
            output_dir: Destination directory.

        Returns:
            A mapping of split name to written path.
        """
        out_dir = Path(output_dir)
        paths = {}
        for name in SPLIT_NAMES:
            paths[name] = self.write_jsonl(splits.get(name, []), out_dir / f"{name}.jsonl")
        return paths

    def write_dataset_manifest(
        self,
        splits: dict[str, list[dict[str, Any]]],
        output_path: Path | str,
    ) -> Path:
        """Write a per-example manifest (id, split, task_type, source).

        Args:
            splits: The split mapping.
            output_path: Destination JSONL path.

        Returns:
            The path written to.
        """
        rows: list[dict[str, Any]] = []
        for name, examples in splits.items():
            for example in examples:
                rows.append(
                    {
                        "id": example["id"],
                        "split": name,
                        "task_type": example["task_type"],
                        "section": example.get("metadata", {}).get("section", ""),
                        "ticker": example.get("metadata", {}).get("ticker", ""),
                        "source": example.get("source", ""),
                    }
                )
        return self.write_jsonl(rows, output_path)

    # ------------------------------------------------------------------
    # Stats + leakage
    # ------------------------------------------------------------------
    def check_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Check for company (ticker) overlap between splits.

        Args:
            splits: The split mapping.

        Returns:
            A dict with the three pairwise overlap lists and a ``passed`` flag
            (``True`` when train and test share no company).
        """
        tickers = {
            name: {self._company_key(ex) for ex in examples} for name, examples in splits.items()
        }
        train_test = sorted(tickers["train"] & tickers["test"])
        train_val = sorted(tickers["train"] & tickers["validation"])
        val_test = sorted(tickers["validation"] & tickers["test"])
        return {
            "train_test_ticker_overlap": train_test,
            "train_validation_ticker_overlap": train_val,
            "validation_test_ticker_overlap": val_test,
            "passed": len(train_test) == 0,
        }

    def write_dataset_stats(
        self,
        splits: dict[str, list[dict[str, Any]]],
        output_path: Path | str,
    ) -> Path:
        """Compute and write dataset statistics as JSON.

        Args:
            splits: The split mapping.
            output_path: Destination JSON path.

        Returns:
            The path written to.
        """
        all_examples = [ex for examples in splits.values() for ex in examples]
        input_lengths = [len(ex["input"]) for ex in all_examples] or [0]
        output_lengths = [len(ex["output"]) for ex in all_examples] or [0]

        stats: dict[str, Any] = {
            "total_examples": len(all_examples),
            "examples_per_split": {name: len(ex) for name, ex in splits.items()},
            "examples_per_task_type": dict(Counter(ex["task_type"] for ex in all_examples)),
            "examples_per_section": dict(
                Counter(ex.get("metadata", {}).get("section", "") for ex in all_examples)
            ),
            "average_input_length": round(sum(input_lengths) / len(input_lengths), 2),
            "average_output_length": round(sum(output_lengths) / len(output_lengths), 2),
            "max_input_length": max(input_lengths),
            "min_input_length": min(input_lengths),
            "per_split": {},
            "leakage_check": self.check_leakage(splits),
        }

        for name, examples in splits.items():
            tickers = sorted({self._company_key(ex) for ex in examples})
            ciks = sorted({str(ex.get("metadata", {}).get("cik", "")) for ex in examples} - {""})
            years = sorted({str(ex.get("metadata", {}).get("year", "")) for ex in examples} - {""})
            stats["per_split"][name] = {
                "examples": len(examples),
                "unique_tickers": tickers,
                "unique_ciks": ciks,
                "year_range": [years[0], years[-1]] if years else [],
            }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote dataset stats to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def build(
        self,
        max_examples: int | None = None,
        train_ratio: float = 0.85,
        validation_ratio: float = 0.10,
        test_ratio: float = 0.05,
        strategy: str = "company_holdout",
        task_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline: build, split, and write all outputs.

        Args:
            max_examples: Optional cap on total examples.
            train_ratio: Target train fraction.
            validation_ratio: Target validation fraction.
            test_ratio: Target test fraction.
            strategy: Split strategy.
            task_types: Optional task-type whitelist.

        Returns:
            A summary dict with output paths, split sizes, and the leakage check.
        """
        examples = self.build_all_examples(max_examples=max_examples, task_types=task_types)
        splits = self.split_examples(
            examples,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            strategy=strategy,
        )
        paths = self.write_dataset(splits, self.output_dir)
        stats_path = self.write_dataset_stats(splits, self.output_dir / "dataset_stats.json")
        manifest_path = self.write_dataset_manifest(
            splits, self.output_dir / "dataset_manifest.jsonl"
        )
        return {
            "paths": {k: str(v) for k, v in paths.items()},
            "stats_path": str(stats_path),
            "manifest_path": str(manifest_path),
            "split_sizes": {name: len(ex) for name, ex in splits.items()},
            "leakage_check": self.check_leakage(splits),
            "task_types": sorted({ex["task_type"] for ex in examples}),
            "expected_task_types": TASK_TYPES,
        }
