"""Final repository-readiness check for FinSage-7B (Phase 12).

Validates that the repo is portfolio/publish ready and — critically — *honest*:
required docs exist, no secrets or model weights are committed, and if the
benchmark report contains mock/sample numbers the README clearly says so.

The logic lives in :func:`run_checks` (pure, takes a ``root`` path) so it is unit
testable without touching the real repo. The CLI renders a Rich table and exits
non-zero only when a **critical** check fails.

Example::

    python scripts/final_repo_check.py
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finsage.logging_utils import get_logger, setup_logging

app = typer.Typer(help="Validate the repository for portfolio release.", add_completion=False)
logger = get_logger(__name__)
console = Console()

#: File extensions that indicate committed model weights.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".ckpt", ".pth")

#: Secret-like config keys to scan for hardcoded real values.
_SECRET_KEYS = (
    "API_SECRET_KEY",
    "SECRET_KEY",
    "HF_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "WANDB_API_KEY",
)

#: Values that are acceptable placeholders (never treated as a real secret).
_PLACEHOLDER_HINTS = (
    "change-me",
    "changeme",
    "your-",
    "your_",
    "xxx",
    "placeholder",
    "example",
    "dummy",
    "redacted",
    "<",
    "${",
    "...",
    "none",
    "test",
)

#: File types worth scanning for secrets (code/config, not prose).
_SECRET_SCAN_SUFFIXES = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".env",
)

#: Marker substring identifying a mock/sample benchmark report.
_MOCK_MARKER = "Sample/mock report for pipeline validation only"

#: README phrases that count as an honest mock-vs-real disclosure.
_MOCK_WARNING_HINTS = (
    "pending gpu",
    "sample pipeline-validation",
    "not real performance",
    "not real benchmark",
    "not model performance",
    "validate the evaluation pipeline only",
    "validate the reporting pipeline only",
    "sample/mock",
)


@dataclass
class Check:
    """Result of a single repository check.

    Attributes:
        name: Short human-readable check name.
        passed: Whether the check passed.
        critical: Whether a failure should cause a non-zero exit.
        detail: Optional explanation shown in the report.
    """

    name: str
    passed: bool
    critical: bool = False
    detail: str = ""


def _tracked_files(root: Path) -> list[Path]:
    """Return repo-tracked files (via git), or all files if not a git repo.

    Args:
        root: Repository root.

    Returns:
        A list of paths (relative to ``root``) that are committed/tracked. When
        ``root`` is not a git work tree, every non-VCS file under ``root`` is
        returned as a conservative fallback.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        files = [line for line in out.stdout.splitlines() if line.strip()]
        if files:
            return [Path(f) for f in files]
    except (subprocess.SubprocessError, OSError):
        pass
    skip = {".git", ".venv", "venv", "node_modules", ".next", "__pycache__"}
    return [
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and not any(part in skip for part in p.relative_to(root).parts)
    ]


def _looks_like_real_secret(value: str) -> bool:
    """Heuristically decide whether a config value is a real secret.

    Args:
        value: The raw assigned value (without surrounding quotes).

    Returns:
        ``True`` if the value looks like a real credential rather than a
        placeholder/env reference.
    """
    v = value.strip().strip("\"'").strip()
    if len(v) < 8:
        return False
    lowered = v.lower()
    if any(hint in lowered for hint in _PLACEHOLDER_HINTS):
        return False
    # Code/env references (process.env.X, os.environ, import.meta.env, getenv,
    # dotted attribute access) are not literal secrets.
    if "." in v or "env" in lowered or "process" in lowered or "getenv" in lowered:
        return False
    if v.startswith("$") or v.upper() == v and "_" in v and not any(c.isdigit() for c in v):
        # Looks like an env-var reference / config token, not a literal secret.
        return False
    # A real secret tends to mix letters and digits or be a long opaque token.
    has_alpha = any(c.isalpha() for c in v)
    has_digit = any(c.isdigit() for c in v)
    return (has_alpha and has_digit) or len(v) >= 24


def _scan_for_secrets(root: Path, tracked: list[Path]) -> list[str]:
    """Scan tracked code/config files for hardcoded real secrets.

    Args:
        root: Repository root.
        tracked: Tracked files (relative paths).

    Returns:
        A list of ``"path: KEY"`` strings for suspected real secrets.
    """
    key_re = re.compile(
        r"(" + "|".join(_SECRET_KEYS) + r")\s*[:=]\s*([\"']?[^\s\"'#,)]+[\"']?)",
    )
    self_name = Path(__file__).name
    findings: list[str] = []
    for rel in tracked:
        if rel.suffix.lower() not in _SECRET_SCAN_SUFFIXES:
            continue
        parts = rel.parts
        if rel.name == self_name or "tests" in parts or rel.name.endswith(".env.example"):
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in key_re.finditer(text):
            if _looks_like_real_secret(match.group(2)):
                findings.append(f"{rel.as_posix()}: {match.group(1)}")
    return findings


def _file_check(root: Path, rel: str, *, critical: bool = False) -> Check:
    """Build a Check for the existence of a file."""
    exists = (root / rel).is_file()
    return Check(
        name=f"{rel} exists",
        passed=exists,
        critical=critical,
        detail="" if exists else "missing",
    )


def run_checks(root: Path | str = ".") -> list[Check]:
    """Run all repository-readiness checks against ``root``.

    Args:
        root: Repository root to validate.

    Returns:
        A list of :class:`Check` results.
    """
    root = Path(root)
    checks: list[Check] = []

    # --- README content -----------------------------------------------------
    readme = root / "README.md"
    readme_text = readme.read_text(encoding="utf-8", errors="ignore") if readme.is_file() else ""
    lowered = readme_text.lower()
    checks.append(Check("README.md exists", bool(readme_text), critical=True))
    has_disclaimer = "licensed financial advisor" in lowered or "not financial advice" in lowered
    checks.append(
        Check(
            "README has financial disclaimer",
            has_disclaimer,
            critical=True,
            detail=(
                "" if has_disclaimer else "add the 'not a licensed financial advisor' disclaimer"
            ),
        )
    )
    # Match a Markdown heading for "Limitations", tolerating numbered headings
    # like "## 15. Limitations".
    has_limitations = bool(re.search(r"(?m)^#{1,6}\s+(?:\d+[.)]\s*)?limitations\b", lowered))
    checks.append(
        Check(
            "README has limitations section",
            has_limitations,
            critical=True,
            detail="" if has_limitations else "add a Limitations section",
        )
    )

    # --- Honesty: mock vs real ---------------------------------------------
    report = root / "reports" / "benchmark_report.md"
    metadata = root / "reports" / "report_metadata.json"
    report_text = report.read_text(encoding="utf-8", errors="ignore") if report.is_file() else ""
    mock_detected = _MOCK_MARKER in report_text or (
        metadata.is_file() and '"is_sample_report": true' in metadata.read_text("utf-8", "ignore")
    )
    if mock_detected:
        has_warning = any(hint in lowered for hint in _MOCK_WARNING_HINTS)
        checks.append(
            Check(
                "README discloses mock/sample benchmark status",
                has_warning,
                critical=True,
                detail=(
                    ""
                    if has_warning
                    else "mock report detected; README must say numbers are sample/pending"
                ),
            )
        )
    else:
        checks.append(
            Check("No mock report (or real results present)", True, detail="no mock marker")
        )

    # --- Required community / publishing files ------------------------------
    for rel in (
        "CONTRIBUTING.md",
        "SECURITY.md",
        "LICENSE",
        "docs/project_summary.md",
        "docs/interview_guide.md",
        "docs/reproducibility.md",
        "docs/publishing_guide.md",
        "docs/model_card.md",
        "docs/dataset_card.md",
        "reports/final_release_checklist.md",
        "reports/resume_bullets.md",
        "reports/linkedin_post.md",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/pull_request_template.md",
    ):
        checks.append(_file_check(root, rel))

    # benchmark report is desirable but only a warning if absent.
    checks.append(
        Check(
            "reports/benchmark_report.md exists",
            report.is_file(),
            detail="" if report.is_file() else "run 'make report' to generate",
        )
    )

    # --- Security: nothing sensitive committed ------------------------------
    tracked = _tracked_files(root)
    tracked_posix = {p.as_posix() for p in tracked}

    env_committed = ".env" in tracked_posix or "frontend/.env.local" in tracked_posix
    checks.append(
        Check(
            ".env / frontend/.env.local not committed",
            not env_committed,
            critical=True,
            detail="" if not env_committed else "remove committed env file",
        )
    )

    weights = [p.as_posix() for p in tracked if p.suffix.lower() in _WEIGHT_SUFFIXES]
    checks.append(
        Check(
            "No model weights committed",
            not weights,
            critical=True,
            detail="" if not weights else f"weight files tracked: {weights[:3]}",
        )
    )

    checkpoints = [
        p.as_posix()
        for p in tracked
        if p.parts and p.parts[0] == "checkpoints" and p.name != ".gitkeep"
    ]
    checks.append(
        Check(
            "No checkpoint files committed",
            not checkpoints,
            critical=True,
            detail="" if not checkpoints else f"checkpoint files tracked: {checkpoints[:3]}",
        )
    )

    raw_data = [
        p.as_posix()
        for p in tracked
        if p.as_posix().startswith(("data/raw/", "data/processed/")) and p.name != ".gitkeep"
    ]
    checks.append(
        Check(
            "No raw/processed data committed",
            not raw_data,
            critical=True,
            detail="" if not raw_data else f"data files tracked: {raw_data[:3]}",
        )
    )

    secrets = _scan_for_secrets(root, tracked)
    checks.append(
        Check(
            "No hardcoded real secrets",
            not secrets,
            critical=True,
            detail="" if not secrets else f"suspected secrets: {secrets[:3]}",
        )
    )

    return checks


def render(checks: list[Check]) -> bool:
    """Render the checks as a Rich table and return overall pass/fail.

    Args:
        checks: The check results.

    Returns:
        ``True`` if no *critical* check failed.
    """
    table = Table(title="FinSage-7B — Final Repo Check", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail", overflow="fold")
    for c in checks:
        if c.passed:
            mark = "[green]PASS[/green]"
        elif c.critical:
            mark = "[red]FAIL[/red]"
        else:
            mark = "[yellow]WARN[/yellow]"
        table.add_row(c.name, mark, c.detail)
    console.print(table)

    critical_failures = [c for c in checks if c.critical and not c.passed]
    warnings = [c for c in checks if not c.critical and not c.passed]
    if warnings:
        console.print(f"[yellow]{len(warnings)} non-critical warning(s).[/yellow]")
    if critical_failures:
        console.print(f"[red]{len(critical_failures)} critical failure(s).[/red]")
        return False
    console.print("[green]All critical checks passed.[/green]")
    return True


@app.callback(invoke_without_command=True)
def main(
    root: str = typer.Option(".", help="Repository root to check."),
) -> None:
    """Run the final repository check and exit non-zero on critical failure.

    Args:
        root: Repository root to validate.

    Raises:
        typer.Exit: Code 1 if any critical check fails.
    """
    setup_logging("INFO")
    checks = run_checks(root)
    if not render(checks):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
