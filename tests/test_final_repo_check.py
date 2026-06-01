"""Tests for the final repository-readiness checker."""

from __future__ import annotations

from pathlib import Path

from scripts.final_repo_check import app, run_checks
from typer.testing import CliRunner

runner = CliRunner()

_README_OK = (
    "# FinSage-7B\n\n"
    "## Limitations\n\n"
    "- Real benchmark results pending GPU execution.\n\n"
    "## Disclaimer\n\n"
    "FinSage-7B is not a licensed financial advisor. Sample/mock numbers are "
    "sample pipeline-validation results, not real performance.\n"
)


def _check(checks, name_fragment: str):
    """Return the first check whose name contains ``name_fragment``."""
    return next(c for c in checks if name_fragment.lower() in c.name.lower())


def _minimal_repo(tmp_path: Path) -> Path:
    """Create a minimal repo that passes all critical checks."""
    (tmp_path / "README.md").write_text(_README_OK, encoding="utf-8")
    return tmp_path


def test_help_works():
    """`--help` exits cleanly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_passes_on_minimal_repo(tmp_path):
    """A minimal honest repo has no critical failures."""
    checks = run_checks(_minimal_repo(tmp_path))
    critical_failures = [c for c in checks if c.critical and not c.passed]
    assert critical_failures == [], [c.name for c in critical_failures]
    # CLI should also exit 0.
    result = runner.invoke(app, ["--root", str(tmp_path)])
    assert result.exit_code == 0


def test_detects_missing_disclaimer(tmp_path):
    """A README without the disclaimer fails the disclaimer check (critically)."""
    (tmp_path / "README.md").write_text("# FinSage-7B\n\n## Limitations\n\nnone\n", "utf-8")
    checks = run_checks(tmp_path)
    disclaimer = _check(checks, "financial disclaimer")
    assert disclaimer.passed is False and disclaimer.critical is True
    result = runner.invoke(app, ["--root", str(tmp_path)])
    assert result.exit_code == 1


def test_detects_missing_security(tmp_path):
    """A repo without SECURITY.md fails the SECURITY.md existence check."""
    checks = run_checks(_minimal_repo(tmp_path))
    assert _check(checks, "SECURITY.md").passed is False


def test_detects_committed_checkpoint(tmp_path):
    """A committed checkpoint/weight file is flagged."""
    _minimal_repo(tmp_path)
    ckpt = tmp_path / "checkpoints" / "finsage-7b"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_text("BINARY", encoding="utf-8")
    checks = run_checks(tmp_path)
    assert _check(checks, "No model weights committed").passed is False
    assert _check(checks, "No checkpoint files committed").passed is False


def test_detects_real_secret(tmp_path):
    """A hardcoded real-looking secret is flagged."""
    _minimal_repo(tmp_path)
    (tmp_path / "config_local.py").write_text(
        'API_SECRET_KEY = "sk-live-9f8a7b6c5d4e3f2a1b0c"\n', encoding="utf-8"
    )
    secrets_check = _check(run_checks(tmp_path), "hardcoded real secrets")
    assert secrets_check.passed is False and secrets_check.critical is True


def test_allows_placeholder_secret(tmp_path):
    """The 'change-me' placeholder is not treated as a real secret."""
    _minimal_repo(tmp_path)
    (tmp_path / "config_local.py").write_text(
        'API_SECRET_KEY = "change-me"\nHF_TOKEN = "your-token-here"\n', encoding="utf-8"
    )
    assert _check(run_checks(tmp_path), "hardcoded real secrets").passed is True


def test_detects_mock_without_readme_warning(tmp_path):
    """A mock report without a README disclosure fails the honesty check."""
    (tmp_path / "README.md").write_text(
        "# FinSage-7B\n\n## Limitations\n\nnone\n\nnot a licensed financial advisor\n", "utf-8"
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "benchmark_report.md").write_text(
        "Sample/mock report for pipeline validation only. Not real benchmark results.\n", "utf-8"
    )
    honesty = _check(run_checks(tmp_path), "discloses mock")
    assert honesty.passed is False and honesty.critical is True
