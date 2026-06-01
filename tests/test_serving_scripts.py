"""Tests for serving CLI help and the vLLM launch script (no real server)."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.check_vllm_server import app as check_app
from serving.benchmark_latency import app as benchmark_app
from serving.test_endpoint import app as endpoint_app
from typer.testing import CliRunner

runner = CliRunner()
SERVER_SCRIPT = Path("serving/vllm_server.sh")


def test_test_endpoint_help():
    """serving/test_endpoint.py --help renders and lists subcommands."""
    result = runner.invoke(endpoint_app, ["--help"])
    assert result.exit_code == 0
    assert "health" in result.output and "chat" in result.output


def test_benchmark_help():
    """serving/benchmark_latency.py --help renders."""
    result = runner.invoke(benchmark_app, ["--help"])
    assert result.exit_code == 0
    assert "base-url" in result.output


def test_check_vllm_help():
    """scripts/check_vllm_server.py --help renders."""
    result = runner.invoke(check_app, ["--help"])
    assert result.exit_code == 0
    assert "base-url" in result.output


def test_server_script_exists_and_executable():
    """The launch script exists and is marked executable."""
    assert SERVER_SCRIPT.exists()
    assert os.access(SERVER_SCRIPT, os.X_OK)


def test_server_script_contents():
    """The launch script invokes vllm serve and validates MODEL_PATH."""
    content = SERVER_SCRIPT.read_text(encoding="utf-8")
    assert "vllm" in content and "serve" in content
    assert "MODEL_PATH" in content
    assert "not found" in content  # missing-model guard
    assert "set -euo pipefail" in content
