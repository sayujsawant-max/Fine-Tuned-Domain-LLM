"""Convenience check for a running vLLM server (Phase 7).

Runs a health check and a chat smoke test via :class:`VLLMClient`.

Example::

    python scripts/check_vllm_server.py --base-url http://localhost:8000/v1 --model finsage-7b
"""

from __future__ import annotations

import typer
from rich.console import Console

from finsage.logging_utils import get_logger, setup_logging
from finsage.serving.vllm_client import VLLMClient, VLLMClientError

app = typer.Typer(help="Check a running vLLM server.", add_completion=False)
logger = get_logger(__name__)
console = Console()


@app.command()
def main(
    base_url: str = typer.Option("http://localhost:8000/v1", help="vLLM base URL."),
    model: str = typer.Option("finsage-7b", help="Served model name."),
    api_key: str = typer.Option("", help="Optional API key."),
) -> None:
    """Run a health check and a chat smoke test against the vLLM server.

    Args:
        base_url: vLLM base URL.
        model: Served model name.
        api_key: Optional API key.

    Raises:
        typer.Exit: With code 1 if the health check or chat test fails.
    """
    setup_logging("INFO")
    client = VLLMClient(base_url=base_url, model=model, api_key=api_key or None)
    try:
        health = client.health()
        models = [entry.get("id") for entry in health.get("data", [])]
        console.print(f"[green]Health OK[/green] — models: {models}")
        text = client.chat_text("Summarize: the company faces competition and supply chain risk.")
    except VLLMClientError as exc:
        console.print(f"[red]vLLM check failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not text.strip():
        console.print("[red]Chat returned empty content.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Chat OK[/green] — {len(text)} chars returned.")


if __name__ == "__main__":
    app()
