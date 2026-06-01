# Dataset Card — FinSage-7B Instruction Dataset

> ⚠️ **Dataset card template.** Build the dataset locally with the provided CLIs;
> only redistribute derived text if you are comfortable with SEC source terms.

## Dataset name

FinSage-7B SEC-filing instruction dataset.

## Dataset description

Instruction-tuning examples for analysing U.S. SEC filings, derived from public
EDGAR filings. Each example pairs a task instruction + filing excerpt (+ optional
question) with a target answer.

## Source data

Public U.S. SEC EDGAR filings (10-K / 10-Q / 8-K). U.S. government filings are
public; respect EDGAR’s fair-access policy and rate limits (the ingestion client
caps at ~5 req/s and sets a descriptive User-Agent).

## Data processing

1. Download filing submissions/documents (cached).
2. Clean HTML and detect Item-heading sections (risk factors, MD&A, business, …).
3. Render deterministic instruction targets per task type (template/extractive).
4. Split and validate (leakage check, duplicate-id check, task coverage).

## Task types

`risk_summary`, `mda_explanation`, `metric_extraction`, `yoy_comparison`,
`business_risk_identification`, `revenue_driver_explanation`, `filing_qa`,
`analyst_summary`, `outlook_classification`, `hallucination_detection`.

## Splits

Train / validation / test, written with per-split statistics to
`dataset_stats.json`.

## Leakage prevention

Examples are partitioned by **company** (and optionally by **time**) so no company
in training appears in the test set. A leakage check is computed and asserted;
`validate_dataset.py` fails on train/test company overlap.

## Intended use

Fine-tuning and evaluating domain LLMs for filing analysis; research and
education.

## Not intended use

Investment decisions; redistribution that violates source terms; any use implying
the targets are authoritative ground truth (they are weak supervision).

## Limitations

- **Weak supervision:** targets are template/extractive (no LLM teacher), so they
  approximate rather than certify correct answers.
- Coverage depends on which filings/sections were ingested.
- English U.S. filings only.

## License / source terms

Derived from public SEC EDGAR data (U.S. government, public domain). Pipeline code
is Apache-2.0. Verify redistribution terms before publishing derived text.

## Disclaimer

Not financial advice. Derived summaries must be verified against the original
filing.
