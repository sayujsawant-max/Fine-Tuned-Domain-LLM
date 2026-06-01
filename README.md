# FinSage-7B

> A fine-tuned domain LLM for financial filing analysis — adapting Mistral-7B to SEC filings with QLoRA, rigorous before/after evaluation, and production serving.

**Current status: Phase 7 — vLLM serving ✅** (client/health/benchmark + tests are CPU-only; serving the model needs a GPU)

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
| 3 | Chunking + instruction dataset generation (JSONL) | ✅ Done |
| 4 | Baseline evaluation (base Mistral-7B, mock + real backends) | ✅ Done |
| 5 | QLoRA fine-tuning pipeline (dry-run + real) | ✅ Done |
| 6 | Fine-tuned evaluation + before/after benchmark | ✅ Done |
| 7 | vLLM OpenAI-compatible serving | ✅ Done |
| 8 | FastAPI backend (auth, logging, disclaimer) | ⏳ Planned |
| 9 | Frontend demo | ⏳ Planned |
| 10 | Docker + deployment | ⏳ Planned |
| 11 | Benchmark PDF + Hugging Face publishing | ⏳ Planned |

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

## Instruction dataset (Phase 3)

Turn processed sections into a validated, leakage-safe JSONL instruction
dataset. Deterministic and CPU-only — **Phase 3 uses no LLM/GPT/Claude APIs.**

> ⚠️ **Phase 3 outputs are template/extractive weak-supervision targets**, not
> human-written gold answers. Every example is flagged in its metadata
> (`generation_method: template_extractive`, `weak_supervision: true`). Phase 4
> replaces these with reviewed / LLM-assisted targets.

**The 10 task types:** `risk_summary`, `mda_explanation`, `metric_extraction`,
`yoy_comparison`, `business_risk_identification`, `revenue_driver_explanation`,
`filing_qa`, `analyst_summary`, `outlook_classification`,
`hallucination_detection`. Task types are chosen per section.

**Build:**

```bash
python scripts/build_instruction_dataset.py build \
  --processed-manifest-path data/processed/sec/manifest.jsonl \
  --output-dir data/datasets \
  --split-strategy company_holdout \
  --max-examples 10000
```

**Validate:**

```bash
python scripts/validate_dataset.py validate \
  --train-path data/datasets/train.jsonl \
  --validation-path data/datasets/validation.jsonl \
  --test-path data/datasets/test.jsonl \
  --report-path data/datasets/validation_report.json
```

`make build-dataset` / `make validate-dataset` run the same with defaults.

**Outputs:** `train.jsonl`, `validation.jsonl`, `test.jsonl`,
`dataset_stats.json`, `dataset_manifest.jsonl` (all under `data/datasets/`,
git-ignored).

**Example JSONL row:**

```json
{
  "id": "AAPL-2022-10-K-000108-mda-0-yoy_comparison",
  "instruction": "Identify any year-over-year comparison or period-over-period change mentioned in the filing excerpt.",
  "input": "Net revenue for fiscal 2022 was $394,328 million, an increase of 8% compared with the prior year. ...",
  "output": "- Net revenue for fiscal 2022 was $394,328 million, an increase of 8% compared with the prior year.",
  "task_type": "yoy_comparison",
  "source": "AAPL 2022 10-K mda",
  "metadata": {"ticker": "AAPL", "year": "2022", "form": "10-K", "section": "mda",
               "generation_method": "template_extractive", "weak_supervision": true}
}
```

**Leakage-safe splits (`company_holdout`).** Whole companies are assigned to
train/validation/test so **no ticker appears in both train and test**. A
`time_holdout` strategy (latest filing years → test) is also available. The
validator enforces no duplicate ids and no train/test company overlap.

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

## QLoRA fine-tuning (Phase 5)

Fine-tune the base model on the Phase 3 instruction dataset using **QLoRA**:
the base model is frozen and loaded in 4-bit (NF4), and only a small **LoRA
adapter** is trained. The adapter is saved separately from the base weights —
it's a few MB, cheap to version and share, and merged into the base only for
deployment.

> **Train on a GPU.** Mistral-7B QLoRA needs a CUDA GPU (a single 24 GB
> RTX 3090/4090 works; A100 80 GB is faster). It will not run usefully on CPU.

**Dry-run (CPU, no model, no heavy deps)** — validates files/configs/schema and
previews SFT formatting:

```bash
python training/train.py \
  --train-file tests/fixtures/train_sample.jsonl \
  --validation-file tests/fixtures/validation_sample.jsonl \
  --output-dir /tmp/finsage_train_dry_run \
  --config configs/training_config.yaml \
  --lora-config configs/lora_config.yaml \
  --dry-run
```

`make train-dry-run` runs exactly this.

**Real training (GPU):**

```bash
pip install -e ".[ml,training]"
wandb login   # optional; or use --report-to none

python training/train.py \
  --train-file data/datasets/train.jsonl \
  --validation-file data/datasets/validation.jsonl \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --output-dir checkpoints/finsage-7b \
  --config configs/training_config.yaml \
  --lora-config configs/lora_config.yaml \
  --use-4bit --report-to wandb
```

**Resume from a checkpoint:**

```bash
python training/train.py ... --resume-from-checkpoint checkpoints/finsage-7b/checkpoint-1200
```

**Merge the adapter** into the base model for serving:

```bash
python training/merge_adapter.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter-path checkpoints/finsage-7b \
  --output-dir checkpoints/finsage-7b-merged
```

**Outputs** (in `--output-dir`, git-ignored): the LoRA adapter, the tokenizer,
and `training_summary.json` (examples, LoRA r/alpha, lr, epochs, final
train/eval loss). W&B tracking is optional (`--report-to wandb|none`). See
[docs/training_guide.md](docs/training_guide.md).

## Baseline evaluation (Phase 4)

**What it is:** running the *un-fine-tuned* base model over the held-out test set
and scoring it, to establish the "before" numbers. **Why before fine-tuning:**
without a baseline the post-fine-tune scores mean nothing — the whole project
thesis is the *delta* between base Mistral-7B and FinSage-7B on identical data
and metrics.

Two backends:

- **`mock`** (default, CPU-only, no dependencies) — deterministic answers for
  exercising the full pipeline and CI. **It only validates plumbing; its scores
  are not a real model baseline.**
- **`transformers`** (optional, GPU) — real Hugging Face inference on the base
  model. Requires `pip install -e ".[ml,training]"`.

**Mock run:**

```bash
python evaluation/run_baseline_eval.py \
  --test-file tests/fixtures/eval_test_sample.jsonl \
  --output-dir /tmp/finsage_baseline_eval \
  --backend mock --max-examples 20
```

**Real run:**

```bash
pip install -e ".[ml,training]"
python evaluation/run_baseline_eval.py \
  --test-file data/datasets/test.jsonl \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --output-dir reports/figures \
  --backend transformers --device auto --load-in-4bit --max-examples 200
```

`make eval-baseline` (mock) / `make eval-baseline-real` (GPU) run the defaults.

**Outputs** (in `reports/figures/`, plus the Markdown report):
`baseline_predictions.jsonl`, `baseline_results.json`,
`baseline_metrics_by_task.json`, and
[reports/baseline_eval_report.md](reports/baseline_eval_report.md).

**Per-task metrics:** exact match / token F1 (QA), ROUGE-L (summary/explanation
tasks), numeric match (metric extraction), classification accuracy
(outlook/hallucination), and a lightweight `lexical_faithfulness` proxy (a real
NLI/LLM-judge metric replaces it later). See
[docs/eval_guide.md](docs/eval_guide.md).

## Fine-tuned evaluation & benchmark (Phase 6)

Evaluate the fine-tuned model on **the same held-out test set** as the baseline
(identical prompts + metrics — that's what makes the before/after delta valid),
then generate the comparison artifacts and a publishable benchmark report.

Three backends: **`mock`** (CPU/CI), **`adapter`** (base + LoRA adapter, GPU),
**`merged`** (merged model, GPU).

> ⚠️ **Mock results are for pipeline validation only — not real benchmark
> numbers.** Run the `adapter`/`merged` backend on a GPU for headline figures.

**Mock run:**

```bash
python evaluation/run_finetuned_eval.py \
  --test-file tests/fixtures/eval_test_sample.jsonl \
  --baseline-results tests/fixtures/baseline_results_sample.json \
  --baseline-predictions tests/fixtures/baseline_predictions_sample.jsonl \
  --output-dir /tmp/finsage_finetuned_eval --backend mock --max-examples 20
```

**Adapter / merged (GPU):**

```bash
python evaluation/run_finetuned_eval.py \
  --test-file data/datasets/test.jsonl \
  --baseline-results reports/figures/baseline_results.json \
  --baseline-predictions reports/figures/baseline_predictions.jsonl \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter-path checkpoints/finsage-7b \
  --output-dir reports/figures --backend adapter --device auto --load-in-4bit --max-examples 200
# merged: drop --model-id/--adapter-path and pass --merged-model-path checkpoints/finsage-7b-merged --backend merged
```

**Compare already-generated outputs** (no re-inference):

```bash
python evaluation/compare_models.py \
  --baseline-results reports/figures/baseline_results.json \
  --baseline-predictions reports/figures/baseline_predictions.jsonl \
  --finetuned-results reports/figures/finetuned_results.json \
  --finetuned-predictions reports/figures/finetuned_predictions.jsonl \
  --output-dir reports/figures --report-path reports/benchmark_report.md
```

`make eval-finetuned` (mock), `eval-finetuned-adapter`, `eval-finetuned-merged`,
and `compare-models` wrap these.

**Outputs** (`reports/figures/`, git-ignored): `finetuned_predictions.jsonl`,
`finetuned_results.json`, `finetuned_metrics_by_task.json`,
`comparison_results.json`, `metric_delta_by_task.json`,
`comparison_summary.json`, `qualitative_comparisons.jsonl`, optional PNG charts —
plus the committed [reports/benchmark_report.md](reports/benchmark_report.md).
The report has an executive summary, overall + per-task delta tables, best
improvements, regressions, side-by-side qualitative examples, and the disclaimer.

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

## vLLM serving (Phase 7)

Serve the merged FinSage-7B as an **OpenAI-compatible** API with vLLM. This phase
is the inference engine only — the public FastAPI wrapper (auth, logging,
disclaimer) is **Phase 8**.

> **GPU required.** vLLM needs CUDA; it won't run on CPU/macOS/Windows. The
> client, health, and benchmark tooling (and all tests) are CPU-only.
>
> ⚠️ **Do not expose the vLLM port publicly** — it has no auth. Phase 8 adds the
> wrapper.

**1. Install + ensure a merged model exists:**

```bash
pip install -e ".[serving]"
# Merge the trained adapter if you haven't already:
python training/merge_adapter.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter-path checkpoints/finsage-7b \
  --output-dir checkpoints/finsage-7b-merged
```

**2. Start the server** (`make serve-vllm`):

```bash
MODEL_PATH=checkpoints/finsage-7b-merged SERVED_MODEL_NAME=finsage-7b \
  bash serving/vllm_server.sh
```

**3. Test the endpoint** (`make test-vllm`):

```bash
python serving/test_endpoint.py all --base-url http://localhost:8000/v1 --model finsage-7b
# raw curl:
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "finsage-7b",
  "messages": [{"role":"user","content":"Summarize the key risk factors: competition, supply chain, regulation."}],
  "temperature": 0.0, "max_tokens": 256 }'
```

**4. Benchmark latency** (`make benchmark-vllm`):

```bash
python serving/benchmark_latency.py --base-url http://localhost:8000/v1 \
  --model finsage-7b --num-requests 20 --concurrency 1 \
  --output-path reports/figures/vllm_latency_benchmark.json
```

**Docker (GPU + NVIDIA Container Toolkit):**

```bash
docker build -f docker/Dockerfile.serving -t finsage-vllm:latest .
docker compose -f docker/docker-compose.yml up vllm
```

**Troubleshooting:** CUDA OOM → lower `GPU_MEMORY_UTILIZATION`/`MAX_MODEL_LEN`;
`Merged model not found` → run `merge_adapter.py` or set `MODEL_PATH`; `vllm not
found` → `pip install -e ".[serving]"`; gated base model → set `HF_TOKEN`; port
8000 in use → change `VLLM_PORT`; missing chat template → use the merged model
(it ships the tokenizer); slow startup → 7B weights take time to load (poll
`/v1/models`). See [docs/deployment_guide.md](docs/deployment_guide.md).

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
