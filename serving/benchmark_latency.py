"""Lightweight latency benchmark for the vLLM chat-completions endpoint (Phase 7).

Sends concurrent chat-completion requests and reports latency percentiles and
approximate throughput. Uses ``asyncio`` + ``httpx.AsyncClient``; a custom
transport can be injected for fast, network-free unit tests.

Example::

    python serving/benchmark_latency.py --base-url http://localhost:8000/v1 \\
        --model finsage-7b --num-requests 20 --concurrency 1 \\
        --output-path reports/figures/vllm_latency_benchmark.json
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import typer

from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Benchmark vLLM chat-completion latency.", add_completion=False)
logger = get_logger(__name__)

DEFAULT_PROMPT = (
    "Summarize the key risk factors in this filing excerpt: The company faces "
    "competition, supply chain disruption, inflation, and regulatory uncertainty. "
    "Demand softened year over year while costs rose."
)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile of a sorted list (0.0 if empty).

    Args:
        sorted_values: Ascending-sorted values.
        pct: Percentile in ``[0, 100]``.

    Returns:
        The percentile value (nearest-rank), or ``0.0`` for an empty list.
    """
    if not sorted_values:
        return 0.0
    rank = max(0, min(len(sorted_values) - 1, int(round((pct / 100.0) * len(sorted_values)) - 1)))
    return round(sorted_values[rank], 4)


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> tuple[bool, float, int]:
    """Send one chat-completion request and time it.

    Args:
        client: The async HTTP client.
        url: The chat-completions URL.
        payload: The request body.
        headers: Request headers.
        semaphore: Concurrency limiter.

    Returns:
        A ``(success, latency_seconds, completion_tokens)`` tuple.
    """
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(url, json=payload, headers=headers)
            elapsed = time.perf_counter() - start
            if response.status_code != 200:
                return False, elapsed, 0
            data = response.json()
            tokens = int(data.get("usage", {}).get("completion_tokens", 0) or 0)
            return True, elapsed, tokens
        except (httpx.HTTPError, ValueError):
            return False, time.perf_counter() - start, 0


async def _run_async(
    base_url: str,
    model: str,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    api_key: str | None,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any]:
    """Run the benchmark coroutine and aggregate results."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    semaphore = asyncio.Semaphore(max(1, concurrency))

    wall_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0, transport=transport) as client:
        results = await asyncio.gather(
            *(_one_request(client, url, payload, headers, semaphore) for _ in range(num_requests))
        )
    wall_elapsed = time.perf_counter() - wall_start

    latencies = sorted(lat for ok, lat, _ in results if ok)
    successful = len(latencies)
    failed = num_requests - successful
    total_tokens = sum(tok for ok, _, tok in results if ok)

    return {
        "base_url": base_url,
        "model": model,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "successful_requests": successful,
        "failed_requests": failed,
        "p50_latency_s": _percentile(latencies, 50),
        "p95_latency_s": _percentile(latencies, 95),
        "p99_latency_s": _percentile(latencies, 99),
        "avg_latency_s": round(sum(latencies) / successful, 4) if successful else 0.0,
        "min_latency_s": round(latencies[0], 4) if successful else 0.0,
        "max_latency_s": round(latencies[-1], 4) if successful else 0.0,
        "total_wall_time_s": round(wall_elapsed, 4),
        "total_completion_tokens": total_tokens,
        "approx_tokens_per_second": (
            round(total_tokens / wall_elapsed, 2) if wall_elapsed > 0 and total_tokens else 0.0
        ),
    }


def run_benchmark(
    base_url: str,
    model: str,
    num_requests: int = 20,
    concurrency: int = 1,
    max_tokens: int = 256,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Run the latency benchmark synchronously.

    Args:
        base_url: vLLM base URL.
        model: Served model name.
        num_requests: Total requests to send.
        concurrency: Maximum in-flight requests.
        max_tokens: Max tokens per request.
        api_key: Optional API key.
        transport: Optional httpx transport (injected by tests for mocking).

    Returns:
        A JSON-serialisable metrics dict (no NaN/Infinity).
    """
    return asyncio.run(
        _run_async(base_url, model, num_requests, concurrency, max_tokens, api_key, transport)
    )


@app.command()
def main(
    base_url: str = typer.Option("http://localhost:8000/v1", help="vLLM base URL."),
    model: str = typer.Option("finsage-7b", help="Served model name."),
    num_requests: int = typer.Option(20, help="Total requests to send."),
    concurrency: int = typer.Option(1, help="Maximum in-flight requests."),
    max_tokens: int = typer.Option(256, help="Max tokens per request."),
    api_key: str = typer.Option("", help="Optional API key."),
    output_path: str = typer.Option(
        "reports/figures/vllm_latency_benchmark.json", help="JSON output path."
    ),
) -> None:
    """Benchmark the endpoint and write the metrics JSON.

    Args:
        base_url: vLLM base URL.
        model: Served model name.
        num_requests: Total requests to send.
        concurrency: Maximum in-flight requests.
        max_tokens: Max tokens per request.
        api_key: Optional API key.
        output_path: Destination JSON path.

    Raises:
        typer.Exit: With code 1 if all requests fail (endpoint unavailable).
    """
    setup_logging("INFO")
    metrics = run_benchmark(base_url, model, num_requests, concurrency, max_tokens, api_key or None)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "requests=%d ok=%d failed=%d p50=%.3fs p95=%.3fs p99=%.3fs",
        metrics["num_requests"],
        metrics["successful_requests"],
        metrics["failed_requests"],
        metrics["p50_latency_s"],
        metrics["p95_latency_s"],
        metrics["p99_latency_s"],
    )
    logger.info("Wrote latency benchmark to %s", out_path)

    if metrics["successful_requests"] == 0:
        logger.error("All requests failed; is the vLLM server running at %s?", base_url)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
