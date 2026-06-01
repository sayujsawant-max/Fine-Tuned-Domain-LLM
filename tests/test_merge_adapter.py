"""Tests for the merge-adapter CLI (no model downloads)."""

from __future__ import annotations

from training.merge_adapter import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_help():
    """The CLI help renders successfully."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "adapter" in result.output.lower()


def test_missing_adapter_path_clean_error(tmp_path):
    """A missing adapter path errors cleanly before any model is loaded."""
    result = runner.invoke(
        app,
        [
            "--base-model",
            "fake/model",
            "--adapter-path",
            str(tmp_path / "does_not_exist"),
            "--output-dir",
            str(tmp_path / "merged"),
        ],
    )
    assert result.exit_code == 1
    assert "Adapter path not found" in result.output
    # No merged output should have been created (no model download/merge ran).
    assert not (tmp_path / "merged").exists()
