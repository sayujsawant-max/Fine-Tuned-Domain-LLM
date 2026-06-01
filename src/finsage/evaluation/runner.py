"""Evaluation orchestration (Phase 4 baseline).

:class:`EvalRunner` drives a :class:`~finsage.evaluation.generators.BaseGenerator`
over a JSONL test set: it builds prompts, generates and normalises predictions,
computes per-example metrics, aggregates them, and writes the baseline outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finsage.evaluation.generators import BaseGenerator
from finsage.evaluation.metrics import aggregate_metrics, compute_metrics_for_example
from finsage.evaluation.prompts import normalize_prediction
from finsage.logging_utils import get_logger

logger = get_logger(__name__)

_PREVIEW_CHARS = 200


class EvalRunner:
    """Runs a generator over a test set and writes baseline evaluation outputs.

    Args:
        generator: The prediction generator (mock or transformers).
        output_dir: Directory for prediction/metric outputs.
        save_every: Checkpoint predictions to disk every N examples.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        output_dir: Path | str = "reports/figures",
        save_every: int = 25,
    ) -> None:
        self.generator = generator
        self.output_dir = Path(output_dir)
        self.save_every = max(1, save_every)

    def load_examples(
        self, test_file: Path | str, max_examples: int | None = None
    ) -> list[dict[str, Any]]:
        """Load JSONL test examples.

        Args:
            test_file: Path to the JSONL test set.
            max_examples: Optional cap on the number of examples loaded.

        Returns:
            The loaded examples.

        Raises:
            FileNotFoundError: If the test file does not exist.
        """
        path = Path(test_file)
        if not path.exists():
            raise FileNotFoundError(f"Test file not found: {path}")
        examples: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
                if max_examples is not None and len(examples) >= max_examples:
                    break
        logger.info("Loaded %d example(s) from %s", len(examples), path)
        return examples

    def evaluate_examples(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate predictions and compute metrics for each example.

        Args:
            examples: The examples to evaluate.

        Returns:
            Result rows with ``id``, ``task_type``, ``instruction``,
            ``input_preview``, ``reference``, ``prediction``, ``metrics``, and
            ``metadata``.
        """
        rows: list[dict[str, Any]] = []
        for index, example in enumerate(examples, start=1):
            raw = self.generator.generate(example)
            prediction = normalize_prediction(raw)
            metrics = compute_metrics_for_example(example, prediction)
            input_text = str(example.get("input", ""))
            rows.append(
                {
                    "id": example.get("id", f"example-{index}"),
                    "task_type": example.get("task_type", "unknown"),
                    "instruction": example.get("instruction", ""),
                    "input_preview": input_text[:_PREVIEW_CHARS],
                    "reference": example.get("output", ""),
                    "prediction": prediction,
                    "metrics": metrics,
                    "metadata": example.get("metadata", {}),
                }
            )
            if index % self.save_every == 0:
                self.save_predictions(rows, self.output_dir / "_partial_predictions.jsonl")
                logger.info("Checkpointed %d/%d predictions", index, len(examples))
        return rows

    def save_predictions(self, rows: list[dict[str, Any]], path: Path | str) -> Path:
        """Write prediction rows as JSONL.

        Args:
            rows: The result rows.
            path: Destination path.

        Returns:
            The path written to.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out_path

    def save_metrics(self, metrics: dict[str, Any], path: Path | str) -> Path:
        """Write a metrics mapping as JSON.

        Args:
            metrics: The metrics to write.
            path: Destination path.

        Returns:
            The path written to.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    def run_with_prefix(
        self,
        test_file: Path | str,
        output_prefix: str,
        max_examples: int | None = None,
    ) -> dict[str, Any]:
        """Run evaluation and write ``{prefix}_*`` output files.

        Args:
            test_file: Path to the JSONL test set.
            output_prefix: Output file-name prefix (e.g. ``"baseline"`` or
                ``"finetuned"``).
            max_examples: Optional cap on the number of examples.

        Returns:
            The aggregated results dict, augmented with run metadata and the
            output paths.
        """
        examples = self.load_examples(test_file, max_examples=max_examples)
        rows = self.evaluate_examples(examples)
        aggregate = aggregate_metrics(rows)

        results: dict[str, Any] = {
            "backend": getattr(self.generator, "name", type(self.generator).__name__),
            "model_id": getattr(self.generator, "model_id", None),
            "test_file": str(test_file),
            "num_examples": len(rows),
            **aggregate,
        }

        preds_path = self.save_predictions(
            rows, self.output_dir / f"{output_prefix}_predictions.jsonl"
        )
        results_path = self.save_metrics(results, self.output_dir / f"{output_prefix}_results.json")
        by_task_path = self.save_metrics(
            aggregate["by_task"], self.output_dir / f"{output_prefix}_metrics_by_task.json"
        )

        results["paths"] = {
            "predictions": str(preds_path),
            "results": str(results_path),
            "metrics_by_task": str(by_task_path),
        }
        logger.info("%s evaluation complete: %d example(s)", output_prefix, len(rows))
        return results

    def run(self, test_file: Path | str, max_examples: int | None = None) -> dict[str, Any]:
        """Run the full evaluation and write all baseline outputs.

        Args:
            test_file: Path to the JSONL test set.
            max_examples: Optional cap on the number of examples.

        Returns:
            The aggregated results dict (see :meth:`run_with_prefix`).
        """
        return self.run_with_prefix(test_file, "baseline", max_examples=max_examples)
