# FinSage Filing Instruction Dataset

> **Status: Phase 3 (deterministic baseline) + optional Phase 3.5 enhancement.**
> Baseline targets are template/extractive **weak supervision**, not
> human-written gold answers. An optional LLM-assisted pass
> (`scripts/enhance_dataset.py`, the `llm` extra) rewrites them into stronger,
> filing-grounded answers using Claude, keeping the same schema and splits and
> flagging rewritten rows (`generation_method: llm_assisted`,
> `weak_supervision: false`).

## Summary

An instruction-tuning dataset for financial-filing analysis, generated from
**public SEC EDGAR filings** processed in Phase 2 (10-K / 10-Q sections). Built
deterministically with no LLM/GPT/Claude APIs.

## Data source

SEC EDGAR public filings only (see [dataset_guide.md](../docs/dataset_guide.md)).
No private, proprietary, or insider data. Evaluation-only datasets (FinQA,
TAT-QA) are never mixed into training.

## Intended use

- Supervised fine-tuning experiments for filing analysis (QLoRA, Phase 6).
- A reproducible baseline to measure later target-quality improvements against.
- Research and educational use.

## Not intended use

- **Not** investment advice or a basis for financial decisions.
- **Not** a source of verified financial facts — outputs are weakly supervised.
- Not for production deployment without human-reviewed targets and evaluation.

## Task types (10)

`risk_summary`, `mda_explanation`, `metric_extraction`, `yoy_comparison`,
`business_risk_identification`, `revenue_driver_explanation`, `filing_qa`,
`analyst_summary`, `outlook_classification`, `hallucination_detection`. Task
types are selected per section (see the dataset guide).

## Schema (JSONL)

```json
{
  "id": "AAPL-2022-10-K-000108-mda-0-yoy_comparison",
  "instruction": "...",
  "input": "filing excerpt ...",
  "output": "template/extractive target ...",
  "task_type": "yoy_comparison",
  "source": "AAPL 2022 10-K mda",
  "split": "train",
  "metadata": {"ticker": "AAPL", "cik": "0000320193", "form": "10-K",
               "section": "mda", "year": "2022", "chunk_id": 0,
               "generation_method": "template_extractive", "weak_supervision": true}
}
```

## Splits

`train.jsonl`, `validation.jsonl`, `test.jsonl` (+ `dataset_stats.json`,
`dataset_manifest.jsonl`). Target sizes for the full corpus: ~8k–12k train,
~600 validation, ~200–300 test.

## Leakage prevention

Default **`company_holdout`** assigns whole companies to a single split, so **no
company appears in both train and test**. A `time_holdout` (latest years → test)
strategy is also available. A validator enforces no duplicate ids and no
train/test company overlap.

## Known limitations

- Targets are template/extractive weak supervision (see above).
- Extraction is regex/keyword-based and can produce false positives/negatives.
- Outlook labels use naive keyword counting.
- Section extraction quality (Phase 2) bounds dataset quality.

## ⚠️ Financial disclaimer — no investment advice

This dataset and any model trained on it are **not** financial, legal, or
investment advice and produce **no investment recommendations**. Built from
public filings only; always verify against the original source documents.
