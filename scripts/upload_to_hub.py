"""CLI to publish a model/adapter (and card) to the Hugging Face Hub (Phase 5+).

Uploads a local directory — a LoRA adapter, a merged model, the tokenizer, and a
model card — to a Hub repo. ``huggingface_hub`` is imported lazily and this is
**not** run in tests.

Example::

    python scripts/upload_to_hub.py model \\
        --local-dir checkpoints/finsage-7b \\
        --repo-id yourusername/finsage-7b-adapter --private
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Publish artifacts to the Hugging Face Hub.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.callback()
def _main() -> None:
    """Hugging Face Hub uploader (use the ``model`` subcommand)."""


@app.command()
def model(
    local_dir: str = typer.Option(..., "--local-dir", help="Local directory to upload."),
    repo_id: str = typer.Option(..., "--repo-id", help="Target Hub repo id (user/name)."),
    private: bool = typer.Option(False, "--private/--public", help="Create a private repo."),
    repo_type: str = typer.Option("model", help="Hub repo type (model|dataset)."),
    commit_message: str = typer.Option("Upload FinSage-7B artifacts", help="Commit message."),
) -> None:
    """Upload a local model/adapter directory to the Hub.

    Args:
        local_dir: Local directory (adapter or merged model + tokenizer + card).
        repo_id: Target repository id on the Hub.
        private: Whether to create/use a private repo.
        repo_type: Hub repo type.
        commit_message: Commit message for the upload.

    Raises:
        typer.Exit: With code 1 if the directory is missing, deps are absent, or
            ``HF_TOKEN`` is not configured.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    path = Path(local_dir)
    if not path.exists():
        console.print(f"[red]Local directory not found: {local_dir}[/red]")
        raise typer.Exit(code=1)
    if not settings.hf_token:
        console.print("[red]HF_TOKEN is not set; configure it in .env before uploading.[/red]")
        raise typer.Exit(code=1)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        console.print(
            "[red]Uploading requires huggingface_hub. Install with "
            "pip install -e '.[ml,training]'[/red]"
        )
        raise typer.Exit(code=1) from exc

    api = HfApi(token=settings.hf_token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    console.print(f"[green]Uploaded {local_dir} to {repo_id}[/green]")


if __name__ == "__main__":
    app()
