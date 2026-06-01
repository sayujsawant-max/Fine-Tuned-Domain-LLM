# FinSage-7B

> A fine-tuned domain LLM for financial filing analysis — adapting Mistral-7B to SEC filings with QLoRA, rigorous before/after evaluation, and production serving.

**Current status: Phase 2 — SEC EDGAR ingestion & preprocessing ✅** (lightweight, CPU-friendly, no GPU or model downloads required)

---

## What problem it solves

General-purpose LLMs are unreliable on financial filings. They tend to:

- hallucinate financial numbers,
- confuse fiscal years and quarters,
- misuse GAAP terminology,
- summarize filings too generically,
- blur reported facts with forward-looking statements,
- and answer without grounding in the filing text.

**FinSage-7B** adapts an open-source 7B model specifically to SEC filings (10-K, 10-Q, 8-K) so that it answers filing questions, summarizes risk factors, and extracts financial metrics more faithfully than the base model — and proves it with a measurable before/after benchmark.

## Architecture

```
SEC EDGAR
   ↓ ingestion
Section Extraction (Risk Factors / MD&A / Financial Statements)
   ↓ cleaning + chunking
Instruction Dataset Builder (JSONL, 10 task types)
   ↓
Baseline Evaluation (base Mistral-7B)
   ↓
QLoRA Fine-Tuning
   ↓
Fine-Tuned Evaluation  →  Benchmark Report
   ↓
Adapter / Merged Model
   ↓
vLLM (OpenAI-compatible)  →  FastAPI (auth, logging, disclaimer)  →  Frontend Demo
```

> A rendered architecture diagram will live in [`reports/figures/`](reports/figures/). See [docs/architecture.md](docs/architecture.md).

## Project phases

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 1 | Project scaffold, configs, API stubs, tests, tooling | ✅ Done |
| 2 | SEC EDGAR ingestion + section preprocessing | ✅ Done |
| 3 | Chunking + dataset-ready cleaning | ⏳ Planned |
| 4 | Instruction dataset generation (JSONL) | ⏳ Planned |
| 5 | Baseline evaluation (base Mistral-7B) | ⏳ Planned |
| 6 | QLoRA fine-tuning | ⏳ Planned |
| 7 | Fine-tuned evaluation + benchmark report | ⏳ Planned |
| 8 | vLLM serving | ⏳ Planned |
| 9 | FastAPI backend (auth, logging, disclaimer) | ⏳ Planned |
| 10 | Frontend demo | ⏳ Planned |
| 11 | Docker + deployment | ⏳ Planned |
| 12 | Benchmark PDF + Hugging Face publishing | ⏳ Planned |

## Quickstart

This project uses a **`src/` layout** with optional dependency groups so the
default install stays light and CPU-only.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate

# 2. Install the dev toolchain (lightweight)
make install-dev          # or: pip install -e ".[dev]"

# 3. Run the quality gates
make check                # lint + typecheck + test

# 4. Run the API locally (returns a mock response in Phase 1)
make serve-api            # or: uvicorn finsage.serving.app:app --port 8080
curl http://localhost:8080/v1/health
```

> **No `make`?** (e.g. on Windows) every target maps to a plain command —
> see the [Makefile](Makefile) for the exact invocation.

### Optional dependency groups

| Group | Install | Used for |
|-------|---------|----------|
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, black, mypy |
| `ml` | `pip install -e ".[ml]"` | transformers, datasets, evaluate, rouge/bert-score |
| `training` | `pip install -e ".[training]"` | torch, peft, trl, bitsandbytes, accelerate, wandb (**GPU**) |
| `serving` | `pip install -e ".[serving]"` | vLLM (**GPU**) |
| `docs` | `pip install -e ".[docs]"` | mkdocs-material |

Heavy GPU dependencies (`torch`, `bitsandbytes`, `peft`, `vllm`) are **never**
part of the default install.

## SEC ingestion & preprocessing (Phase 2)

Ingest public filings from SEC EDGAR and turn them into clean, section-level
text. CPU-only — no model or GPU dependencies.

**1. Set a descriptive User-Agent** (SEC fair-access rules require one; requests
without it are rejected):

```bash
# macOS/Linux
export EDGAR_USER_AGENT="Your Name your.email@example.com"
# Windows PowerShell
$env:EDGAR_USER_AGENT = "Your Name your.email@example.com"
```

Or add it to your `.env` file (see `.env.example`).

**2. Download filings** → raw HTML + a JSONL manifest:

```bash
python scripts/download_edgar.py download \
  --tickers AAPL MSFT \
  --forms 10-K \
  --start-year 2022 --end-year 2023 \
  --limit-per-company 1
```

**3. Extract sections** → one clean `.txt` per section + a processed manifest:

```bash
python scripts/extract_sections.py extract \
  --manifest-path data/raw/sec/manifest.jsonl \
  --output-dir data/processed/sec \
  --processed-manifest-path data/processed/sec/manifest.jsonl
```

`make download-data` and `make extract-sections` run small defaults.

### Data layout

```
data/raw/sec/{ticker_or_cik}/{form}/{year}/{accession}.html      # raw filings
data/raw/sec/manifest.jsonl                                       # download manifest
data/processed/sec/{ticker_or_cik}/{form}/{year}/{accession}/{section}.txt
data/processed/sec/manifest.jsonl                                 # processed manifest
data/cache/edgar/                                                 # cached SEC JSON
```

Extracted sections: `business`, `risk_factors`, `mda`, `market_risk`,
`financial_statements`.

> **SEC fair access:** the client rate-limits to 5 req/s, retries transient
> errors, and caches metadata. Use **public filings only**, and **do not commit
> raw/processed SEC data** — `data/raw/`, `data/processed/`, and `data/cache/`
> are git-ignored. See [docs/dataset_guide.md](docs/dataset_guide.md).

## Dataset strategy

- **Source:** SEC EDGAR (10-K, 10-Q, 8-K). FinQA / TAT-QA are used for
  evaluation only, never for training.
- **Sections (priority):** Risk Factors (Item 1A), MD&A (Item 7),
  Financial Statements (Item 8), Market Risk (Item 7A), Business (Item 1).
- **Chunking:** ~512 tokens, 64-token overlap, sentence-boundary aware, with
  metadata (`ticker`, `year`, `filing_type`, `section`, `chunk_id`).
- **Splits — no random leakage:** split by *company and year*
  (e.g. train 2018–2022, test 2023, with some companies fully held out).
- Target size: ~8k–12k train / ~600 validation / ~200–300 test examples.

See [docs/dataset_guide.md](docs/dataset_guide.md) and
[data/dataset_card.md](data/dataset_card.md).

## Training plan

- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Method:** QLoRA (4-bit NF4) with PEFT + TRL `SFTTrainer`.
- **LoRA:** rank 16, alpha 32, dropout 0.05 on the attention + MLP projections.
- **Hardware:** fits a single 24 GB GPU (RTX 3090/4090); A100 80 GB for speed.
- Config lives in [configs/lora_config.yaml](configs/lora_config.yaml) and
  [configs/training_config.yaml](configs/training_config.yaml).

See [docs/training_guide.md](docs/training_guide.md).

## Evaluation plan

The before/after benchmark is the heart of the project.

| Task | Primary metric | Secondary |
|------|----------------|-----------|
| Filing QA | Exact Match | Token F1 |
| Risk summary | ROUGE-L | BERTScore |
| Metric extraction | Numeric Exact Match | Unit accuracy |
| Outlook classification | Accuracy | Macro F1 |
| Hallucination | Faithfulness (NLI) | Citation precision |

We compare **base Mistral-7B vs FinSage-7B** (with an optional RAG baseline and
LLM-judge). See [docs/eval_guide.md](docs/eval_guide.md) and
[reports/benchmark_report.md](reports/benchmark_report.md).

## Serving plan

- **vLLM** hosts the merged model as an internal OpenAI-compatible server.
- **FastAPI** wraps it with authentication, request logging, prompt templates,
  safety checks, and mandatory disclaimer injection.

See [docs/deployment_guide.md](docs/deployment_guide.md).

## Deployment plan

Docker Compose orchestrates the stack: `api` (FastAPI) now, with `vllm` and
`frontend` services scaffolded for later phases. See [docker/](docker/).

## ⚠️ Disclaimer — not financial advice

> **FinSage-7B is not a licensed financial advisor. Outputs are not investment
> recommendations.** All responses should be independently verified against the
> original filings. The project uses **public filings only** — no private,
> proprietary, or insider information. The model may hallucinate and has known
> limitations. This is not legal, medical, or financial advice.

## License

Licensed under the [Apache License 2.0](LICENSE).
