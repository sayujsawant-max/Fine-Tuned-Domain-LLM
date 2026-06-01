"""CLI to validate a generated benchmark report (Phase 11).

Checks that the Markdown report exists, is non-empty, contains every required
section (including the limitations and financial disclaimer), carries the
sample/mock label when metadata says it should, has no stray ``TODO`` markers
(unless allowed), and that the metadata file is present and lists missing
artifacts. Exits non-zero on any failure.

Example::

    python scripts/validate_report.py \\
        --report-path reports/benchmark_report.md \\
        --metadata-path reports/report_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from finsage.logging_utils import get_logger, setup_logging
from finsage.reporting.report_builder import (
    DISCLAIMER_HEADING,
    LIMITATIONS_HEADING,
    MOCK_LABEL,
    SECTION_TITLES,
)

app = typer.Typer(help="Validate a generated benchmark report.", add_completion=False)
logger = get_logger(__name__)
console = Console()


def validate_report(
    report_path: Path | str,
    metadata_path: Path | str,
    require_pdf: bool = False,
    allow_todo: bool = False,
) -> list[str]:
    """Validate a report and its metadata, returning a list of failures.

    Args:
        report_path: Path to ``benchmark_report.md``.
        metadata_path: Path to ``report_metadata.json``.
        require_pdf: When ``True`` and metadata requested a PDF, require it.
        allow_todo: When ``True``, do not fail on ``TODO`` markers.

    Returns:
        A list of failure messages (empty when the report is valid).
    """
    failures: list[str] = []
    report = Path(report_path)
    meta = Path(metadata_path)

    if not report.is_file():
        return [f"Report not found: {report}"]
    text = report.read_text(encoding="utf-8")
    if not text.strip():
        failures.append("Report is empty.")

    for title in SECTION_TITLES:
        if f"## {title}" not in text:
            failures.append(f"Missing required section: '{title}'.")
    if DISCLAIMER_HEADING not in text:
        failures.append("Missing financial disclaimer.")
    if f"## {LIMITATIONS_HEADING}" not in text:
        failures.append("Missing limitations section.")
    if not allow_todo and "TODO" in text:
        failures.append("Report contains a 'TODO' marker.")

    metadata: dict = {}
    if not meta.is_file():
        failures.append(f"Metadata not found: {meta}")
    else:
        try:
            metadata = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"Metadata is not valid JSON: {exc}")
        if "missing_artifacts" not in metadata:
            failures.append("Metadata does not list 'missing_artifacts'.")
        if metadata.get("is_sample_report") and MOCK_LABEL not in text:
            failures.append("Metadata flags a sample report but the mock label is missing.")
        if require_pdf and metadata.get("pdf_requested"):
            pdf = report.parent / "benchmark_report.pdf"
            if not pdf.is_file():
                failures.append("PDF was requested but benchmark_report.pdf is missing.")

    return failures


@app.callback(invoke_without_command=True)
def main(
    report_path: str = typer.Option("reports/benchmark_report.md", help="Report Markdown path."),
    metadata_path: str = typer.Option(
        "reports/report_metadata.json", help="Report metadata JSON path."
    ),
    require_pdf: bool = typer.Option(False, help="Require a PDF when one was requested."),
    allow_todo: bool = typer.Option(False, help="Do not fail on TODO markers."),
) -> None:
    """Validate the report and exit non-zero on failure.

    Args:
        report_path: Path to the Markdown report.
        metadata_path: Path to the report metadata JSON.
        require_pdf: Require a PDF if metadata says one was requested.
        allow_todo: Allow ``TODO`` markers in the report.

    Raises:
        typer.Exit: Code 1 if validation fails.
    """
    setup_logging("INFO")
    failures = validate_report(report_path, metadata_path, require_pdf, allow_todo)
    if failures:
        console.print("[red]Report validation FAILED:[/red]")
        for f in failures:
            console.print(f"  • {f}")
        raise typer.Exit(code=1)
    console.print("[green]Report validation PASSED.[/green]")


if __name__ == "__main__":
    app()
