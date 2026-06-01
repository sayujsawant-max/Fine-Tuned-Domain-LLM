"""Convert raw SEC filing HTML into clean, section-level text files.

The preprocessor reads a raw filing, extracts its major sections with
:class:`~finsage.data.section_extractor.SectionExtractor`, writes one ``.txt``
file per section, and emits a JSONL manifest describing every section produced.
No model or GPU dependencies are involved — this is pure CPU text processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finsage.data.section_extractor import SectionExtractor
from finsage.logging_utils import get_logger

logger = get_logger(__name__)


class FilingPreprocessor:
    """Turns raw filing HTML into per-section cleaned text files.

    Args:
        section_extractor: Extractor to use. A default
            :class:`SectionExtractor` is created when ``None``.
    """

    def __init__(self, section_extractor: SectionExtractor | None = None) -> None:
        self.section_extractor = section_extractor or SectionExtractor()

    @staticmethod
    def _identifiers(html_path: Path, metadata: dict[str, Any]) -> dict[str, str]:
        """Resolve path/identifier fields, preferring metadata over the path.

        The raw path is expected to encode
        ``.../{ticker_or_cik}/{form}/{year}/{accession}.html``; those parts are
        used as fallbacks when the corresponding metadata keys are absent.

        Args:
            html_path: Path to the raw HTML filing.
            metadata: Manifest row metadata (may be empty).

        Returns:
            A mapping with ``ticker_or_cik``, ``form``, ``year``, and ``accession``.
        """
        parts = html_path.parts
        path_accession = html_path.stem
        path_year = parts[-2] if len(parts) >= 2 else ""
        path_form = parts[-3] if len(parts) >= 3 else ""
        path_id = parts[-4] if len(parts) >= 4 else ""

        ticker = metadata.get("ticker")
        cik = metadata.get("cik")
        ticker_or_cik = str(ticker or cik or path_id or "UNKNOWN")

        filing_date = str(metadata.get("filing_date", ""))
        year = filing_date[:4] if filing_date[:4].isdigit() else (path_year or "0000")

        return {
            "ticker_or_cik": ticker_or_cik,
            "form": str(metadata.get("form") or path_form or "UNKNOWN").replace("/", "-"),
            "year": year,
            "accession": str(metadata.get("accession_number_no_dashes") or path_accession),
        }

    def process_file(
        self,
        html_path: Path | str,
        output_dir: Path | str = "data/processed/sec",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract sections from one raw filing and write them to disk.

        Args:
            html_path: Path to the raw HTML filing.
            output_dir: Root directory for processed output.
            metadata: Optional manifest row carrying CIK/ticker/form/date fields.

        Returns:
            One manifest row per extracted section, each with ``raw_path``,
            ``processed_path``, ``section``, ``text_chars``, ``text_words``,
            ``cik``, ``ticker``, ``form``, ``filing_date``, ``report_date``,
            ``accession_number``, and ``source_url``.
        """
        html_path = Path(html_path)
        meta = metadata or {}
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        sections = self.section_extractor.extract_sections(html)

        ids = self._identifiers(html_path, meta)
        section_dir = (
            Path(output_dir) / ids["ticker_or_cik"] / ids["form"] / ids["year"] / ids["accession"]
        )

        rows: list[dict[str, Any]] = []
        for section, text in sections.items():
            section_dir.mkdir(parents=True, exist_ok=True)
            processed_path = section_dir / f"{section}.txt"
            processed_path.write_text(text, encoding="utf-8")
            rows.append(
                {
                    "raw_path": str(html_path),
                    "processed_path": str(processed_path),
                    "section": section,
                    "text_chars": len(text),
                    "text_words": len(text.split()),
                    "cik": meta.get("cik", ""),
                    "ticker": meta.get("ticker", ""),
                    "form": meta.get("form", ids["form"]),
                    "filing_date": meta.get("filing_date", ""),
                    "report_date": meta.get("report_date", ""),
                    "accession_number": meta.get("accession_number", ids["accession"]),
                    "source_url": meta.get("document_url", ""),
                }
            )

        logger.info("Processed %s -> %d section(s)", html_path.name, len(rows))
        return rows

    def process_manifest(
        self,
        manifest_path: Path | str,
        output_dir: Path | str = "data/processed/sec",
    ) -> list[dict[str, Any]]:
        """Process every downloaded filing listed in a raw manifest.

        Args:
            manifest_path: Path to the JSONL manifest written by the downloader.
            output_dir: Root directory for processed output.

        Returns:
            The combined processed-manifest rows across all filings.

        Raises:
            FileNotFoundError: If the manifest does not exist.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        all_rows: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                raw_path = row.get("raw_path")
                if not raw_path or not Path(raw_path).exists():
                    logger.warning("Skipping missing raw file: %s", raw_path)
                    continue
                all_rows.extend(self.process_file(raw_path, output_dir=output_dir, metadata=row))

        logger.info("Processed manifest -> %d section row(s)", len(all_rows))
        return all_rows

    @staticmethod
    def write_manifest(rows: list[dict[str, Any]], path: Path | str) -> Path:
        """Write processed-manifest rows as JSONL.

        Args:
            rows: The manifest rows to write.
            path: Destination JSONL path.

        Returns:
            The path written to.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote processed manifest with %d row(s) to %s", len(rows), out_path)
        return out_path
