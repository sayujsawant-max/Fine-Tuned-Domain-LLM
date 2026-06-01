"""QLoRA fine-tuning CLI (Phase 5).

``--dry-run`` validates files, configs, and dataset schema and formats a few
examples **without importing torch/transformers or loading any model**, so it
works on a plain CPU machine. Real training requires ``.[ml,training]`` and a GPU.

Example (real)::

    python training/train.py \\
        --train-file data/datasets/train.jsonl \\
        --validation-file data/datasets/validation.jsonl \\
        --model-id mistralai/Mistral-7B-Instruct-v0.3 \\
        --output-dir checkpoints/finsage-7b \\
        --config configs/training_config.yaml \\
        --lora-config configs/lora_config.yaml \\
        --use-4bit --report-to wandb
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging
from finsage.training.data_formatter import (
    count_tokens_approx,
    format_sft_example,
    validate_training_example,
)
from finsage.training.qlora_trainer import load_yaml_config

app = typer.Typer(help="QLoRA fine-tuning for FinSage-7B.", add_completion=False)
logger = get_logger(__name__)
console = Console()


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        The parsed rows.

    Raises:
        ValueError: If a line is not valid JSON.
    """
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no} invalid JSON: {exc}") from exc
    return rows


def _dry_run(
    train_file: Path,
    validation_file: Path,
    config_path: Path,
    lora_config_path: Path,
    model_id: str,
    output_dir: str,
    use_4bit: bool,
    report_to: str,
) -> None:
    """Validate inputs and preview formatting without loading a model.

    Args:
        train_file: Training JSONL path.
        validation_file: Validation JSONL path.
        config_path: Training config YAML path.
        lora_config_path: LoRA config YAML path.
        model_id: Base model id (reported only).
        output_dir: Output directory (reported only).
        use_4bit: Whether 4-bit would be used (reported only).
        report_to: Reporting backend (reported only).

    Raises:
        typer.Exit: With code 1 if validation fails.
    """
    training_config = load_yaml_config(config_path)
    lora_config = load_yaml_config(lora_config_path)

    train_rows = _read_jsonl(train_file)
    val_rows = _read_jsonl(validation_file)

    errors: list[str] = []
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        for i, row in enumerate(rows):
            for err in validate_training_example(row):
                errors.append(f"{split}[{i}]: {err}")
    if errors:
        for err in errors[:20]:
            console.print(f"[red]{err}[/red]")
        console.print(f"[red]Dry-run found {len(errors)} schema error(s).[/red]")
        raise typer.Exit(code=1)

    preview = format_sft_example(train_rows[0]) if train_rows else "(no examples)"
    avg_tokens = (
        sum(count_tokens_approx(format_sft_example(r)) for r in train_rows) / len(train_rows)
        if train_rows
        else 0
    )

    table = Table(title="QLoRA dry-run summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("model_id", model_id)
    table.add_row("output_dir", output_dir)
    table.add_row("use_4bit", str(use_4bit))
    table.add_row("report_to", report_to)
    table.add_row("train_examples", str(len(train_rows)))
    table.add_row("validation_examples", str(len(val_rows)))
    table.add_row("lora_r / alpha", f"{lora_config.get('r')} / {lora_config.get('lora_alpha')}")
    table.add_row("max_seq_length", str(training_config.get("sft", {}).get("max_seq_length")))
    table.add_row("packing", str(training_config.get("sft", {}).get("packing")))
    table.add_row("avg approx tokens/example", f"{avg_tokens:.0f}")
    console.print(table)
    console.print("[green]Dry-run OK[/green] — schema valid, configs load, formatting works.")
    console.print("First formatted example (truncated):")
    console.print(preview[:400] + ("…" if len(preview) > 400 else ""), markup=False)


@app.command()
def main(
    train_file: str = typer.Option(..., "--train-file", help="Training JSONL path."),
    validation_file: str = typer.Option(..., "--validation-file", help="Validation JSONL path."),
    model_id: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.3", "--model-id", help="Base model id."
    ),
    output_dir: str = typer.Option("checkpoints/finsage-7b", help="Adapter output directory."),
    config: str = typer.Option("configs/training_config.yaml", help="Training config YAML."),
    lora_config: str = typer.Option("configs/lora_config.yaml", help="LoRA config YAML."),
    use_4bit: bool = typer.Option(True, "--use-4bit/--no-use-4bit", help="Use 4-bit QLoRA."),
    resume_from_checkpoint: str | None = typer.Option(None, help="Checkpoint path to resume from."),
    max_train_samples: int | None = typer.Option(None, help="Cap on training examples."),
    max_eval_samples: int | None = typer.Option(None, help="Cap on validation examples."),
    report_to: str = typer.Option("wandb", help="Reporting backend (wandb|none|tensorboard)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only; do not train."),
) -> None:
    """Run (or dry-run) QLoRA fine-tuning.

    Args:
        train_file: Training JSONL path.
        validation_file: Validation JSONL path.
        model_id: Base model id.
        output_dir: Adapter output directory.
        config: Training config YAML path.
        lora_config: LoRA config YAML path.
        use_4bit: Whether to use 4-bit QLoRA loading.
        resume_from_checkpoint: Optional checkpoint to resume from.
        max_train_samples: Optional cap on training examples.
        max_eval_samples: Optional cap on validation examples.
        report_to: Reporting backend.
        dry_run: If set, validate and preview only (no model, no training).

    Raises:
        typer.Exit: With code 1 on missing files or missing training dependencies.
    """
    setup_logging(get_settings().log_level)

    train_path, val_path = Path(train_file), Path(validation_file)
    if not train_path.exists():
        console.print(f"[red]Training file not found: {train_path}[/red]")
        raise typer.Exit(code=1)
    if not val_path.exists():
        console.print(f"[red]Validation file not found: {val_path}[/red]")
        raise typer.Exit(code=1)

    if dry_run:
        try:
            _dry_run(
                train_path,
                val_path,
                Path(config),
                Path(lora_config),
                model_id,
                output_dir,
                use_4bit,
                report_to,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        return

    # Real training — heavy deps imported lazily inside train().
    from finsage.training.qlora_trainer import train as run_training

    if report_to != load_yaml_config(config).get("training", {}).get("report_to"):
        logger.info(
            "Note: --report-to=%s; set 'training.report_to' in %s to change the backend.",
            report_to,
            config,
        )

    try:
        summary = run_training(
            train_file=train_path,
            validation_file=val_path,
            model_id=model_id,
            output_dir=output_dir,
            training_config_path=config,
            lora_config_path=lora_config,
            use_4bit=use_4bit,
            resume_from_checkpoint=resume_from_checkpoint,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # surface a clean message, non-zero exit
        console.print(f"[red]Training failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Training complete.[/green] Summary: {summary}")


if __name__ == "__main__":
    app()
