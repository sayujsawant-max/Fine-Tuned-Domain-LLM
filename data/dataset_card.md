# FinSage-7B Instruction Dataset Card

> **Status: scaffold (Phase 1).** The dataset has not been built yet. This card
> documents the intended design and will be finalised in Phase 4.

## Summary

An instruction-tuning dataset derived from **public SEC EDGAR filings**
(10-K, 10-Q, 8-K) for adapting a 7B LLM to financial-filing analysis.

## Sources

- **Training/eval source:** SEC EDGAR (public filings only).
- **Evaluation-only references:** FinQA, TAT-QA (never used for training).
- No private, proprietary, or insider information is used.

## Sections covered

Risk Factors (Item 1A), MD&A (Item 7), Financial Statements (Item 8),
Market Risk (Item 7A), Business (Item 1).

## Schema (JSONL)

```json
{
  "id": "AAPL-2022-10-K-RISK_SUMMARY-0042",
  "source": "AAPL 2022 10-K Risk Factors",
  "instruction": "Summarize the top three risk factors disclosed in this filing excerpt.",
  "input": "Filing excerpt text ...",
  "output": "The company discloses three key risks: ...",
  "task_type": "risk_summary",
  "metadata": {"ticker": "AAPL", "year": 2022, "filing_type": "10-K", "section": "Risk Factors"}
}
```

## Task types (10)

`risk_summary`, `mda_explanation`, `metric_extraction`, `yoy_comparison`,
`business_risk_identification`, `revenue_driver_explanation`, `filing_qa`,
`analyst_summary`, `outlook_classification`, `hallucination_detection`.

## Splits (no leakage)

Splits are made **by company and year**, not by random example:

| Split | Target size | Policy |
|-------|-------------|--------|
| train | 8,000–12,000 | e.g. 2018–2022 filings |
| validation | ~600 | held-out companies |
| test | 200–300 | e.g. 2023 filings + fully held-out companies |

## Licensing & disclaimer

Built from public-domain U.S. government filings. **Not financial advice.**
Always verify against the original filings.
