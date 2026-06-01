"""Smoke-check a running FinSage-7B FastAPI server (Phase 8).

Exercises ``/health``, ``/ready``, ``/config``, and ``/chat`` and prints a
summary table. Exits non-zero if any check fails.

Example::

    python scripts/check_api_server.py --base-url http://localhost:8080/v1 --api-key change-me
"""

from __future__ import annotations

from typing import Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Check a running FinSage-7B API server.", add_completion=False)
logger = get_logger(__name__)
console = Console()

#: Fake excerpt used for the /chat smoke test (no real filing data).
SAMPLE_EXCERPT = (
    "The company faces supply chain disruption, competition, and regulatory uncertainty."
)


def _check(name: str, func: Any) -> tuple[str, bool, str]:
    """Run a single check, capturing success and a short detail string."""
    try:
        detail = func()
        return name, True, detail
    except httpx.HTTPStatusError as exc:
        return name, False, f"HTTP {exc.response.status_code}"
    except httpx.HTTPError as exc:
        return name, False, f"connection error: {exc}"
    except (KeyError, ValueError) as exc:
        return name, False, f"bad payload: {exc}"


@app.command()
def main(
    base_url: str = typer.Option("http://localhost:8080/v1", help="API base URL."),
    api_key: str = typer.Option("change-me", help="API key for protected endpoints."),
    timeout: float = typer.Option(30.0, help="Per-request timeout (seconds)."),
) -> None:
    """Run health/readiness/config/chat checks against the API.

    Args:
        base_url: API base URL (including the ``/v1`` prefix).
        api_key: API key sent via the ``X-API-Key`` header.
        timeout: Per-request timeout in seconds.

    Raises:
        typer.Exit: With code 1 if any check fails.
    """
    setup_logging("INFO")
    base = base_url.rstrip("/")
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    client = httpx.Client(timeout=timeout, headers=headers)

    def health() -> str:
        resp = client.get(f"{base}/health")
        resp.raise_for_status()
        return str(resp.json()["status"])

    def ready() -> str:
        resp = client.get(f"{base}/ready")
        resp.raise_for_status()
        data = resp.json()
        return f"{data['status']} (vllm_available={data['vllm_available']})"

    def config() -> str:
        resp = client.get(f"{base}/config")
        resp.raise_for_status()
        return f"model={resp.json()['model']}"

    def chat() -> str:
        resp = client.post(
            f"{base}/chat",
            json={
                "question": "Summarize the key risk factors.",
                "filing_excerpt": SAMPLE_EXCERPT,
                "task_type": "risk_summary",
                "max_tokens": 128,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        return f"{len(resp.json()['answer'])} chars"

    results = [
        _check("GET /health", health),
        _check("GET /ready", ready),
        _check("GET /config", config),
        _check("POST /chat", chat),
    ]
    client.close()

    table = Table(title=f"FinSage API check — {base}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, ok, detail in results:
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(name, status, detail)
    console.print(table)

    if not all(ok for _, ok, _ in results):
        console.print("[red]One or more checks failed.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]All checks passed.[/green]")


if __name__ == "__main__":
    app()
