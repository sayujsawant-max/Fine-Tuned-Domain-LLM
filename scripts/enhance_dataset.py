"""CLI to upgrade weak-supervision dataset targets with Claude (Phase 3.5).

Reads a JSONL instruction dataset and rewrites each ``output`` into a stronger,
filing-grounded answer via :class:`LLMTargetGenerator`. Use ``--mock`` for a
deterministic, offline/credential-free run; real generation needs the ``llm``
extra (``pip install -e '.[llm]'``) and ``ANTHROPIC_API_KEY``.

Example::

    python scripts/enhance_dataset.py enhance \\
        --input-path data/datasets/train.jsonl \\
        --output-path data/datasets/train_enhanced.jsonl --mock
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console

from finsage.config import get_settings
from finsage.data.llm_target_generator import DEFAULT_MODEL, LLMTargetGenerator
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="LLM-assisted dataset target enhancement.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.callback()
def _main() -> None:
    """Dataset target enhancer (use the ``enhance`` subcommand)."""


@app.command()
def enhance(
    input_path: str = typer.Option(..., "--input-path", help="Source JSONL dataset."),
    output_path: str = typer.Option(..., "--output-path", help="Enhanced JSONL output."),
    model: str = typer.Option(DEFAULT_MODEL, help="Claude model id (real mode)."),
    mock: bool = typer.Option(False, "--mock/--no-mock", help="Deterministic offline mode."),
    max_examples: int | None = typer.Option(None, help="Cap on examples to enhance."),
    max_tokens: int = typer.Option(1024, help="Max tokens per generation."),
) -> None:
    """Enhance dataset targets and write a new JSONL file.

    Args:
        input_path: Path to the source JSONL dataset.
        output_path: Destination path for the enhanced JSONL.
        model: Claude model id (real mode).
        mock: If set, run deterministically offline with no API calls.
        max_examples: Optional cap on the number of examples.
        max_tokens: Max tokens per generation.

    Raises:
        typer.Exit: With code 1 on a missing input or missing credentials/deps.
    """
    setup_logging(get_settings().log_level)

    src = Path(input_path)
    if not src.exists():
        console.print(f"[red]Input dataset not found: {input_path}[/red]")
        raise typer.Exit(code=1)

    if not mock and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]ANTHROPIC_API_KEY is not set. Use --mock for an offline run, or set "
            "the key and install the LLM extra: pip install -e '.[llm]'[/red]"
        )
        raise typer.Exit(code=1)

    examples: list[dict] = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_examples is not None and len(examples) >= max_examples:
                break

    generator = LLMTargetGenerator(model=model, mock=mock, max_tokens=max_tokens)
    try:
        enhanced = generator.enhance_examples(examples)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for example in enhanced:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    mode = "mock" if mock else f"model={model}"
    console.print(f"[green]Enhanced {len(enhanced)} example(s) ({mode}) -> {output_path}[/green]")


if __name__ == "__main__":
    app()
