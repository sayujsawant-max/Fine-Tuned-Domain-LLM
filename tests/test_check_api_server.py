"""Tests for scripts/check_api_server.py (no real server required)."""

from __future__ import annotations

from scripts.check_api_server import app as check_app
from typer.testing import CliRunner

runner = CliRunner()


def test_check_api_help():
    """The --help output renders and documents options."""
    result = runner.invoke(check_app, ["--help"])
    assert result.exit_code == 0
    assert "base-url" in result.output
    assert "api-key" in result.output


def test_check_api_handles_unavailable_server():
    """Against an unreachable server the script exits non-zero cleanly."""
    result = runner.invoke(
        check_app,
        ["--base-url", "http://127.0.0.1:9/v1", "--api-key", "x", "--timeout", "1"],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output
