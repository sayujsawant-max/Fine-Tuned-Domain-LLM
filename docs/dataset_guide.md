# Dataset Guide

Covers SEC ingestion + preprocessing (Phase 2, implemented) and the
instruction-dataset build that follows (Phases 3–4, planned).

## Data source

All data comes from **SEC EDGAR — public filings only**. We never use private,
proprietary, or insider information. Endpoints:

- Company tickers: `https://www.sec.gov/files/company_tickers.json`
- Submissions:     `https://data.sec.gov/submissions/CIK##########.json`
- Documents:       `https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}`

**SEC fair access.** Every request sends a descriptive `User-Agent` from
`EDGAR_USER_AGENT`. The client rate-limits to 5 req/s, retries `429/5xx` with
backoff, and caches the JSON metadata under `data/cache/edgar/` so it is never
re-fetched. Do not scrape aggressively.

## Pipeline (Phase 2)

```
download_edgar.py download  → data/raw/sec/...           + raw manifest
extract_sections.py extract → data/processed/sec/...     + processed manifest
```

### 1. Download

```bash
export EDGAR_USER_AGENT="Your Name your.email@example.com"
python scripts/download_edgar.py download \
  --tickers AAPL MSFT --forms 10-K 10-Q \
  --start-year 2021 --end-year 2023 --limit-per-company 5
```

Resolves each ticker to its CIK, lists filings (filtered by form and filing
year, newest first), and downloads each primary document. Existing files are
skipped unless `--force`.

### 2. Extract sections

```bash
python scripts/extract_sections.py extract \
  --manifest-path data/raw/sec/manifest.jsonl \
  --output-dir data/processed/sec \
  --processed-manifest-path data/processed/sec/manifest.jsonl
```

Cleans HTML to text and extracts five sections via robust `Item`-heading
detection (which skips short table-of-contents entries): `business` (Item 1),
`risk_factors` (1A), `mda` (7), `market_risk` (7A), `financial_statements` (8).
Missing sections are skipped, not fatal.

## Storage formats

### Raw filings

```
data/raw/sec/{ticker_or_cik}/{form}/{year}/{accession_no_dashes}.html
data/raw/sec/manifest.jsonl
```

### Processed sections

```
data/processed/sec/{ticker_or_cik}/{form}/{year}/{accession_no_dashes}/{section}.txt
data/processed/sec/manifest.jsonl
```

## Manifest fields

**Raw manifest** (one row per filing) — from `EdgarClient`:

| Field | Meaning |
|-------|---------|
| `cik` | 10-digit zero-padded CIK |
| `ticker` | Ticker symbol (when known) |
| `accession_number` / `accession_number_no_dashes` | EDGAR accession id |
| `filing_date` / `report_date` | ISO dates |
| `form` | Form type (`10-K`, `10-Q`, …) |
| `primary_document` | Primary document file name |
| `filing_url` / `document_url` | Index page / primary document URL |
| `raw_path` / `downloaded` | Local path and success flag |

**Processed manifest** (one row per extracted section) — from `FilingPreprocessor`:

`raw_path`, `processed_path`, `section`, `text_chars`, `text_words`, `cik`,
`ticker`, `form`, `filing_date`, `report_date`, `accession_number`, `source_url`.

## ⚠️ Leakage warning (for the future train/test split)

When the instruction dataset is built (Phase 4), **split by company and year, not
by random example** — e.g. train on 2018–2022, test on 2023, and hold some
companies out entirely. Splitting raw chunks at random leaks near-duplicate text
(boilerplate risk factors recur across years/filers) and inflates metrics. Track
`ticker` + `year` through every stage so the split can be enforced.

## Do not commit SEC data

`data/raw/`, `data/processed/`, and `data/cache/` (and `*.html` / `*.htm`) are
git-ignored. Keep filings out of the repo — they are large and reproducible from
the manifests. Only `.gitkeep` placeholders, the dataset card, and small test
fixtures under `tests/fixtures/` are tracked.

## Instruction dataset generation (Phase 3)

Processed sections become a validated JSONL instruction dataset.
**Deterministic and CPU-only — no LLM/GPT/Claude APIs are used in this phase.**

```bash
make build-dataset      # or: python scripts/build_instruction_dataset.py build ...
make validate-dataset   # or: python scripts/validate_dataset.py validate ...
```

### Instruction dataset format

One JSON object per line with fields: `id`, `instruction`, `input`, `output`,
`task_type`, `source`, `metadata` (and `split` once assigned). `metadata`
carries `ticker`, `cik`, `form`, `section`, `year`, `accession_number`,
`chunk_id`, plus the weak-supervision flags `generation_method` and
`weak_supervision`.

### Task type definitions

| Task type | Applied to sections | Output |
|-----------|---------------------|--------|
| `risk_summary` | risk_factors | extractive summary (first 2–4 sentences) |
| `mda_explanation` | mda | extractive summary |
| `metric_extraction` | financial_statements, mda, market_risk | regex `$`/`%`/magnitude/bps bullets |
| `yoy_comparison` | mda, financial_statements | sentences with comparison language |
| `business_risk_identification` | business, risk_factors | sentences with risk keywords |
| `revenue_driver_explanation` | mda, business | sentences with revenue keywords |
| `filing_qa` | all | extractive answer (first 1–3 sentences) |
| `analyst_summary` | all | structured Summary / Key Point / Evidence |
| `outlook_classification` | mda, business, risk_factors | `{label, reason}` via keyword scoring |
| `hallucination_detection` | risk_factors, market_risk | `{supported, reason}` (supported vs. generic claim) |

Where a regex/keyword search finds nothing, the renderer emits an explicit
"No explicit … was found" message so every output is non-empty.

### Chunking strategy

Whitespace tokenization, `max_tokens=512`, `overlap=64`, `min_tokens=80`. Short
trailing windows are dropped unless they are the only chunk; metadata is copied
onto every chunk; empty chunks are never produced.

### Splitting & leakage prevention

- **`company_holdout`** (default): whole companies are assigned to
  train/validation/test so **no company appears in both train and test**.
- **`time_holdout`**: latest filing years go to test.

Allocation guarantees train keeps ≥1 company and (with enough companies)
validation and test each get ≥1. Splitting is deterministic for a given
`random_seed`.

### Validation rules

`scripts/validate_dataset.py` checks: every example parses as JSON; required
fields present and non-null; non-empty `input`/`output`; valid `task_type` and
`split`; **no duplicate ids across splits**; **no train/test company overlap**;
and task-type coverage. It exits non-zero on failure and writes a JSON report.

### Dataset statistics

`dataset_stats.json` reports totals, examples per split / task type / section,
average / min / max input and output lengths, unique tickers and CIKs and the
year range per split, and the leakage-check result.

### ⚠️ Limitations of template outputs

Phase 3 targets are **template/extractive weak supervision**, not gold answers.
Extractive summaries can miss the most salient sentence; keyword/regex
extraction has false positives/negatives; outlook labels are naive keyword
counts. They are a reproducible *starting* signal, not ground truth.

### Future improvement path

Phase 4 will upgrade targets via human review and/or LLM-assisted generation
(with a held-out judge), keeping the same schema, splits, and validator so
quality can be compared against this deterministic baseline.
