"""Smoke-test the running FinSage-7B API from the command line.

Unlike the unit tests (which use the in-process TestClient), this hits a live
HTTP endpoint. Run it after ``make serve-api``::

    python serving/test_endpoint.py --base-url http://localhost:8080
"""

from __future__ import annotations

import httpx
import typer

from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Smoke-test the running FinSage-7B API.", add_completion=False)
logger = get_logger(__name__)


@app.command()
def main(
    base_url: str = typer.Option("http://localhost:8080", help="API base URL."),
) -> None:
    """Call ``/v1/health`` and ``/v1/chat`` against a running service.

    Args:
        base_url: Base URL of the running FinSage-7B API.

    Raises:
        typer.Exit: With code 1 if either endpoint does not return HTTP 200.
    """
    setup_logging("INFO")
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        health = client.get("/v1/health")
        logger.info("GET /v1/health -> %d %s", health.status_code, health.json())

        chat = client.post(
            "/v1/chat",
            json={"question": "What are the top risk factors?", "context": "..."},
        )
        logger.info("POST /v1/chat -> %d %s", chat.status_code, chat.json())

    if health.status_code != 200 or chat.status_code != 200:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
