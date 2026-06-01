"""Verify the full FinSage-7B stack end to end (Phase 10).

Checks the frontend, the FastAPI wrapper (``/health``, ``/ready``, ``/chat``),
and — unless ``--demo`` — the internal vLLM server. Prints a rich summary table
and optionally writes a JSON report.

The HTTP client is injectable so unit tests can run without any live services.

Example::

    python scripts/check_full_stack.py \\
        --frontend-url http://localhost:3000 \\
        --api-url http://localhost:8080/v1 \\
        --vllm-url http://localhost:8000/v1 \\
        --api-key change-me
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Verify the FinSage-7B full stack.", add_completion=False)
logger = get_logger(__name__)
console = Console()

#: Fabricated excerpt for the /chat smoke test (no real filing data).
SAMPLE_EXCERPT = (
    "The company faces supply chain disruption, competition, and regulatory uncertainty."
)


def _result(name: str, ok: bool, detail: str) -> dict[str, Any]:
    """Build one check-result record."""
    return {"name": name, "ok": ok, "detail": detail}


def run_checks(
    frontend_url: str,
    api_url: str,
    vllm_url: str,
    api_key: str,
    *,
    demo_mode: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run all stack checks and return a structured report.

    Args:
        frontend_url: Base URL of the Next.js frontend.
        api_url: Base URL of the FastAPI wrapper (including ``/v1``).
        vllm_url: Base URL of the vLLM server (including ``/v1``).
        api_key: API key for protected API endpoints.
        demo_mode: When ``True``, the vLLM check is skipped.
        client: Optional pre-built HTTP client (injected by tests). When ``None``
            a default client with a 30s timeout is created and closed here.

    Returns:
        A dict with ``checks`` (list of ``{name, ok, detail}``) and ``all_ok``.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    api = api_url.rstrip("/")
    vllm = vllm_url.rstrip("/")
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    checks: list[dict[str, Any]] = []

    def attempt(name: str, fn: Any, required: bool = True) -> None:
        try:
            checks.append(_result(name, True, fn()))
        except httpx.HTTPStatusError as exc:
            checks.append(_result(name, not required, f"HTTP {exc.response.status_code}"))
        except httpx.HTTPError as exc:
            checks.append(_result(name, not required, f"connection error: {exc}"))
        except (KeyError, ValueError, TypeError) as exc:
            checks.append(_result(name, not required, f"bad payload: {exc}"))

    def frontend() -> str:
        resp = http.get(frontend_url)
        resp.raise_for_status()
        return f"HTTP {resp.status_code}"

    def api_health() -> str:
        resp = http.get(f"{api}/health")
        resp.raise_for_status()
        return str(resp.json()["status"])

    def api_ready() -> str:
        resp = http.get(f"{api}/ready", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return f"{data['status']} (vllm_available={data['vllm_available']})"

    def api_chat() -> str:
        resp = http.post(
            f"{api}/chat",
            headers=headers,
            json={
                "question": "Summarize the key risk factors.",
                "filing_excerpt": SAMPLE_EXCERPT,
                "task_type": "risk_summary",
                "max_tokens": 128,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        answer = resp.json()["answer"]
        if not answer:
            raise ValueError("empty answer")
        return f"{len(answer)} chars"

    def vllm_models() -> str:
        resp = http.get(f"{vllm}/models")
        resp.raise_for_status()
        models = [m.get("id") for m in resp.json().get("data", [])]
        return f"models={models}"

    attempt("frontend reachable", frontend)
    attempt("API /v1/health", api_health)
    attempt("API /v1/ready", api_ready)
    attempt("API /v1/chat", api_chat)
    if demo_mode:
        checks.append(_result("vLLM /v1/models", True, "skipped (demo mode)"))
    else:
        attempt("vLLM /v1/models", vllm_models)

    if owns_client:
        http.close()

    return {"checks": checks, "all_ok": all(c["ok"] for c in checks)}


def write_report(report: dict[str, Any], output_path: str) -> Path:
    """Write the report to JSON, creating parent directories.

    Args:
        report: The report dict from :func:`run_checks`.
        output_path: Destination path.

    Returns:
        The path written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@app.command()
def main(
    frontend_url: str = typer.Option("http://localhost:3000", help="Frontend base URL."),
    api_url: str = typer.Option("http://localhost:8080/v1", help="API base URL."),
    vllm_url: str = typer.Option("http://localhost:8000/v1", help="vLLM base URL."),
    api_key: str = typer.Option("change-me", help="API key for protected endpoints."),
    demo: bool = typer.Option(False, "--demo", help="Skip the vLLM check (demo mode)."),
    output_path: str = typer.Option(
        "reports/figures/full_stack_health.json", help="JSON report output path."
    ),
) -> None:
    """Run the full-stack health checks and print a summary table.

    Args:
        frontend_url: Frontend base URL.
        api_url: API base URL (including ``/v1``).
        vllm_url: vLLM base URL (including ``/v1``).
        api_key: API key for protected endpoints.
        demo: When set, the vLLM check is skipped.
        output_path: Where to write the JSON report.

    Raises:
        typer.Exit: With code 1 if any required check fails.
    """
    setup_logging("INFO")
    report = run_checks(frontend_url, api_url, vllm_url, api_key, demo_mode=demo)

    table = Table(title="FinSage-7B full-stack check")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for c in report["checks"]:
        status = "[green]OK[/green]" if c["ok"] else "[red]FAIL[/red]"
        table.add_row(c["name"], status, c["detail"])
    console.print(table)

    path = write_report(report, output_path)
    console.print(f"Report written to [cyan]{path}[/cyan]")

    if not report["all_ok"]:
        console.print("[red]One or more checks failed.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]All checks passed.[/green]")


if __name__ == "__main__":
    app()
