"""Merge a trained LoRA adapter into the base model (Phase 5).

Produces a standalone merged checkpoint for later deployment (vLLM). Heavy
dependencies are imported lazily; this script is **not** run in tests.

Example::

    python training/merge_adapter.py \\
        --base-model mistralai/Mistral-7B-Instruct-v0.3 \\
        --adapter-path checkpoints/finsage-7b \\
        --output-dir checkpoints/finsage-7b-merged \\
        --torch-dtype bfloat16
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Merge a LoRA adapter into the base model.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.command()
def main(
    base_model: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.3", "--base-model", help="Base model id."
    ),
    adapter_path: str = typer.Option(..., "--adapter-path", help="Trained adapter directory."),
    output_dir: str = typer.Option(
        "checkpoints/finsage-7b-merged", "--output-dir", help="Merged model output directory."
    ),
    torch_dtype: str = typer.Option("bfloat16", help="Torch dtype for the merged model."),
) -> None:
    """Merge the adapter into the base model and save the result.

    Args:
        base_model: Base model id.
        adapter_path: Path to the trained LoRA adapter.
        output_dir: Directory to write the merged model + tokenizer.
        torch_dtype: Torch dtype name for loading/saving.

    Raises:
        typer.Exit: With code 1 if the adapter path is missing or deps absent.
    """
    setup_logging(get_settings().log_level)

    if not Path(adapter_path).exists():
        console.print(f"[red]Adapter path not found: {adapter_path}[/red]")
        raise typer.Exit(code=1)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        console.print(
            "[red]Merging requires torch, transformers, and peft. "
            "Install with pip install -e '.[ml,training]'[/red]"
        )
        raise typer.Exit(code=1) from exc

    dtype = getattr(torch, torch_dtype, torch.bfloat16)
    logger.info("Loading base model %s", base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="auto")

    logger.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    logger.info("Merging adapter into base weights")
    model = model.merge_and_unload()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out_dir))
    console.print(f"[green]Merged model saved to {out_dir}[/green]")


if __name__ == "__main__":
    app()
