"""Smoke-test the running vLLM OpenAI-compatible endpoint (Phase 7).

Subcommands: ``health`` (GET /v1/models), ``chat`` (POST /v1/chat/completions),
and ``all`` (both). Uses :class:`VLLMClient` (httpx only — no openai package).

Examples::

    python serving/test_endpoint.py health --base-url http://localhost:8000/v1
    python serving/test_endpoint.py chat --base-url http://localhost:8000/v1 \\
        --model finsage-7b --prompt "Summarize the key risk factors ..."
    python serving/test_endpoint.py all --base-url http://localhost:8000/v1 --model finsage-7b
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from finsage.logging_utils import get_logger, setup_logging
from finsage.serving.vllm_client import VLLMClient, VLLMClientError

app = typer.Typer(help="Smoke-test the vLLM endpoint.", add_completion=False)
logger = get_logger(__name__)
console = Console()

DEFAULT_PROMPT = (
    "Summarize the key risk factors in this filing excerpt: The company faces "
    "competition, supply chain disruption, and regulatory uncertainty."
)


def _client(base_url: str, model: str, api_key: str | None) -> VLLMClient:
    """Build a VLLMClient from CLI options."""
    return VLLMClient(base_url=base_url, model=model, api_key=api_key or None)


@app.command()
def health(
    base_url: str = typer.Option("http://localhost:8000/v1", help="vLLM base URL."),
    model: str = typer.Option("finsage-7b", help="Served model name."),
    api_key: str = typer.Option("", help="Optional API key."),
) -> None:
    """Check the ``/models`` endpoint.

    Args:
        base_url: vLLM base URL.
        model: Served model name.
        api_key: Optional API key.

    Raises:
        typer.Exit: With code 1 if the health check fails.
    """
    setup_logging("INFO")
    try:
        data = _client(base_url, model, api_key).health()
    except VLLMClientError as exc:
        console.print(f"[red]Health check failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    models = [entry.get("id") for entry in data.get("data", [])]
    console.print(f"[green]Healthy[/green] — served models: {models}")


@app.command()
def chat(
    base_url: str = typer.Option("http://localhost:8000/v1", help="vLLM base URL."),
    model: str = typer.Option("finsage-7b", help="Served model name."),
    prompt: str = typer.Option(DEFAULT_PROMPT, help="User prompt."),
    max_tokens: int = typer.Option(256, help="Max tokens to generate."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    api_key: str = typer.Option("", help="Optional API key."),
) -> None:
    """Send a chat completion and validate the response.

    Args:
        base_url: vLLM base URL.
        model: Served model name.
        prompt: User prompt.
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        api_key: Optional API key.

    Raises:
        typer.Exit: With code 1 if the chat request fails or content is empty.
    """
    setup_logging("INFO")
    try:
        text = _client(base_url, model, api_key).chat_text(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
    except VLLMClientError as exc:
        console.print(f"[red]Chat request failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not text.strip():
        console.print("[red]Chat returned empty content.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Chat OK[/green] ({len(text)} chars):")
    console.print(text, markup=False)


@app.command()
def all(
    base_url: str = typer.Option("http://localhost:8000/v1", help="vLLM base URL."),
    model: str = typer.Option("finsage-7b", help="Served model name."),
    api_key: str = typer.Option("", help="Optional API key."),
) -> None:
    """Run both the health and chat smoke tests.

    Args:
        base_url: vLLM base URL.
        model: Served model name.
        api_key: Optional API key.

    Raises:
        typer.Exit: With code 1 if either check fails.
    """
    setup_logging("INFO")
    client = _client(base_url, model, api_key)
    table = Table(title="vLLM smoke test")
    table.add_column("Check")
    table.add_column("Result")

    ok = True
    try:
        data = client.health()
        models = [entry.get("id") for entry in data.get("data", [])]
        table.add_row("health (/models)", f"[green]PASS[/green] {models}")
    except VLLMClientError as exc:
        ok = False
        table.add_row("health (/models)", f"[red]FAIL[/red] {exc}")

    try:
        text = client.chat_text(DEFAULT_PROMPT)
        status = "PASS" if text.strip() else "FAIL (empty)"
        color = "green" if text.strip() else "red"
        ok = ok and bool(text.strip())
        table.add_row("chat (/chat/completions)", f"[{color}]{status}[/{color}] {len(text)} chars")
    except VLLMClientError as exc:
        ok = False
        table.add_row("chat (/chat/completions)", f"[red]FAIL[/red] {exc}")

    console.print(table)
    if not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
