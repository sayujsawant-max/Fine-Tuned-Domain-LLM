"""Tests for the enhance-dataset CLI (mock mode; no real API)."""

from __future__ import annotations

import json

from scripts.enhance_dataset import app
from typer.testing import CliRunner

runner = CliRunner()

ROWS = [
    {
        "id": "a",
        "instruction": "Summarize.",
        "input": "Revenue grew. Margins expanded. Outlook is positive.",
        "output": "Revenue grew.",
        "task_type": "filing_qa",
        "metadata": {"ticker": "ACME", "weak_supervision": True},
    },
    {
        "id": "b",
        "instruction": "Summarize.",
        "input": "Costs rose. Demand softened.",
        "output": "Costs rose.",
        "task_type": "risk_summary",
        "metadata": {"ticker": "GLOBEX", "weak_supervision": True},
    },
]


def test_help():
    """The CLI help renders."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "enhance" in result.output


def test_mock_enhance_writes_output(tmp_path):
    """Mock enhancement writes enhanced rows flagged as not weak supervision."""
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"

    result = runner.invoke(
        app, ["enhance", "--input-path", str(src), "--output-path", str(out), "--mock"]
    )
    assert result.exit_code == 0, result.output

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert row["metadata"]["enhanced"] is True
        assert row["metadata"]["weak_supervision"] is False
        assert row["output"].strip()


def test_missing_api_key_real_mode_errors(tmp_path, monkeypatch):
    """Real mode without ANTHROPIC_API_KEY exits cleanly with guidance."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(ROWS[0]) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "enhance",
            "--input-path",
            str(src),
            "--output-path",
            str(tmp_path / "o.jsonl"),
            "--no-mock",
        ],
    )
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_missing_input_errors(tmp_path):
    """A missing input dataset errors cleanly."""
    result = runner.invoke(
        app,
        [
            "enhance",
            "--input-path",
            str(tmp_path / "nope.jsonl"),
            "--output-path",
            str(tmp_path / "o.jsonl"),
            "--mock",
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.output
