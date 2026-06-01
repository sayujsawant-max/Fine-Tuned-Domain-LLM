"""CLI to publish the model/adapter and dataset to the Hugging Face Hub (Phase 12)."""

from __future__ import annotations

import typer

from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Publish artifacts to the Hugging Face Hub.", add_completion=False)
logger = get_logger(__name__)


@app.command()
def run(
    repo_id: str = typer.Option(..., help="Target Hub repo id, e.g. 'user/finsage-7b'."),
    what: str = typer.Option(
        "adapter", help="Artifact to upload: 'adapter', 'merged', or 'dataset'."
    ),
) -> None:
    """Upload the selected artifact to the Hugging Face Hub.

    Args:
        repo_id: Target repository id on the Hub.
        what: Which artifact to upload.

    Raises:
        NotImplementedError: Always, until Phase 12. Requires ``HF_TOKEN`` and
            the ``ml`` extras (``huggingface_hub``).
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Phase 12 stub: would upload %s to %s", what, repo_id)
    if not settings.hf_token:
        logger.warning("HF_TOKEN is not set; configure it in .env before uploading.")
    raise NotImplementedError("Hub upload lands in Phase 12.")


if __name__ == "__main__":
    app()
