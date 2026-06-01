"""Optional HTML and PDF exporters for the benchmark report.

Both exporters are best-effort: HTML falls back to a minimal manual wrapper if
the ``markdown`` package is missing, and PDF tries WeasyPrint, then ReportLab,
and otherwise writes ``PDF_EXPORT_SKIPPED.md`` explaining how to enable it. A
failure here never fails the overall report build.
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<link rel="stylesheet" href="report_assets/style.css" />
</head>
<body>
<main class="report">
{body}
</main>
</body>
</html>
"""


def _module_available(name: str) -> bool:
    """Return ``True`` if a module can be imported."""
    import importlib.util

    return importlib.util.find_spec(name) is not None


def check_pdf_export_available() -> dict[str, bool]:
    """Report which optional export backends are importable.

    Returns:
        A mapping with boolean availability for ``markdown``, ``weasyprint``,
        ``reportlab`` and ``pandoc``.
    """
    import shutil

    return {
        "markdown": _module_available("markdown"),
        "weasyprint": _module_available("weasyprint"),
        "reportlab": _module_available("reportlab"),
        "pandoc": shutil.which("pandoc") is not None,
    }


def export_markdown_to_html(markdown_path: Path | str, html_path: Path | str) -> Path | None:
    """Convert a Markdown report to HTML.

    Uses the ``markdown`` package when available (with tables/fenced-code
    extensions); otherwise wraps the raw Markdown in a ``<pre>`` block so the
    export still succeeds.

    Args:
        markdown_path: Source Markdown file.
        html_path: Destination HTML file.

    Returns:
        The HTML path on success, else ``None``.
    """
    src = Path(markdown_path)
    if not src.is_file():
        logger.warning("Markdown source missing for HTML export: %s", src)
        return None
    text = src.read_text(encoding="utf-8")

    try:
        import markdown as md  # type: ignore[import-untyped]

        body = md.markdown(text, extensions=["tables", "fenced_code", "toc"])
    except Exception as exc:  # pragma: no cover - depends on optional dep
        logger.warning("markdown package unavailable (%s); using <pre> fallback.", exc)
        body = f"<pre>{html_lib.escape(text)}</pre>"

    out = Path(html_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_HTML_SHELL.format(title="FinSage-7B Benchmark Report", body=body), "utf-8")
    logger.info("Wrote HTML report to %s", out)
    return out


def _pdf_via_weasyprint(markdown_path: Path, pdf_path: Path) -> Path | None:
    """Render Markdown -> HTML -> PDF via WeasyPrint, or ``None`` on failure."""
    try:
        import markdown as md  # type: ignore[import-untyped]
        from weasyprint import HTML  # type: ignore[import-untyped]

        body = md.markdown(
            markdown_path.read_text(encoding="utf-8"),
            extensions=["tables", "fenced_code"],
        )
        doc = _HTML_SHELL.format(title="FinSage-7B Benchmark Report", body=body)
        HTML(string=doc, base_url=str(markdown_path.parent)).write_pdf(str(pdf_path))
        logger.info("Wrote PDF report via WeasyPrint to %s", pdf_path)
        return pdf_path
    except Exception as exc:  # pragma: no cover - optional/native deps
        logger.warning("WeasyPrint PDF export failed: %s", exc)
        return None


def _pdf_via_reportlab(markdown_path: Path, pdf_path: Path) -> Path | None:
    """Render a basic but readable PDF from Markdown text via ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
        from reportlab.lib.styles import getSampleStyleSheet  # type: ignore[import-untyped]
        from reportlab.platypus import (  # type: ignore[import-untyped]
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )

        styles = getSampleStyleSheet()
        story = []
        for raw in markdown_path.read_text(encoding="utf-8").splitlines():
            line = raw.rstrip()
            if not line:
                story.append(Spacer(1, 6))
                continue
            if line.startswith("### "):
                story.append(Paragraph(html_lib.escape(line[4:]), styles["Heading3"]))
            elif line.startswith("## "):
                story.append(Paragraph(html_lib.escape(line[3:]), styles["Heading2"]))
            elif line.startswith("# "):
                story.append(Paragraph(html_lib.escape(line[2:]), styles["Heading1"]))
            else:
                story.append(Paragraph(html_lib.escape(line), styles["BodyText"]))
        SimpleDocTemplate(str(pdf_path), pagesize=letter).build(story)
        logger.info("Wrote PDF report via ReportLab to %s", pdf_path)
        return pdf_path
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("ReportLab PDF export failed: %s", exc)
        return None


_SKIP_NOTE = """# PDF export skipped

The benchmark report Markdown was generated successfully, but no PDF backend was
available, so `{pdf_name}` was not produced.

To enable PDF export, install one of:

```bash
pip install -e ".[reporting]"   # installs reportlab (cross-platform)
# or, for higher-fidelity output on Linux/macOS:
pip install weasyprint
# or install pandoc and re-run report generation
```

Then re-run:

```bash
make report
```
"""


def export_markdown_to_pdf(markdown_path: Path | str, pdf_path: Path | str) -> Path | None:
    """Export the Markdown report to PDF, degrading gracefully.

    Tries WeasyPrint, then ReportLab. If neither is available, writes a
    ``PDF_EXPORT_SKIPPED.md`` note next to the target PDF and returns ``None``.
    Never raises.

    Args:
        markdown_path: Source Markdown file.
        pdf_path: Destination PDF file.

    Returns:
        The PDF path on success, else ``None``.
    """
    src = Path(markdown_path)
    out = Path(pdf_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        logger.warning("Markdown source missing for PDF export: %s", src)
        return None

    for backend in (_pdf_via_weasyprint, _pdf_via_reportlab):
        result = backend(src, out)
        if result is not None:
            return result

    note = out.parent / "PDF_EXPORT_SKIPPED.md"
    note.write_text(_SKIP_NOTE.format(pdf_name=out.name), encoding="utf-8")
    logger.warning("No PDF backend available; wrote %s", note)
    return None
