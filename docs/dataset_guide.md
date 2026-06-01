# Dataset Guide (Phases 2–4)

## Pipeline

```
download_edgar.py  → data/raw/        (raw filing HTML)
extract_sections.py → data/processed/ (Risk Factors / MD&A / Financials)
build_instruction_dataset.py → data/datasets/{train,validation,test}.jsonl
validate_dataset.py → schema + task-type checks
```

## 1. Download (Phase 2)

```bash
python scripts/download_edgar.py run --cik 0000320193 --form-type 10-K --limit 5
```

SEC EDGAR requires a descriptive `User-Agent` with contact info — set
`EDGAR_USER_AGENT` in `.env`. Respect the ~10 req/s fair-access limit.

## 2. Extract sections (Phase 3)

```bash
python scripts/extract_sections.py run --raw-dir data/raw --out-dir data/processed
```

Priority sections: Risk Factors (1A), MD&A (7), Financial Statements (8),
Market Risk (7A), Business (1).

## 3. Chunk + build instructions (Phase 4)

- Chunk to ~512 tokens with 64-token overlap, preserving
  `ticker`, `year`, `filing_type`, `section`, `chunk_id`.
- Emit one of the 10 task types per example (see `finsage.data.instruction_builder`).
- **Targets (`output`) come from a teacher model or filing-grounded extraction —
  never hand-fabricated.**

```bash
python scripts/build_instruction_dataset.py run
python scripts/validate_dataset.py run
```

## 4. Splitting (no leakage)

Split by **company and year**, not by random example. See
[../data/dataset_card.md](../data/dataset_card.md).
