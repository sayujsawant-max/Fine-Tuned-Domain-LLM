# FinSage-7B

**Fine-tuned domain LLM pipeline for financial filing analysis** — built on SEC
filings, QLoRA, Mistral-7B, vLLM, FastAPI, and Next.js.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Next.js](https://img.shields.io/badge/frontend-Next.js%2014-black)
![Docker](https://img.shields.io/badge/deploy-Docker%20Compose-2496ED)
[![CI](https://github.com/sayujsawant-max/Fine-Tuned-Domain-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/sayujsawant-max/Fine-Tuned-Domain-LLM/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

> ⚠️ **Status — honesty note:** the full pipeline is implemented and tested
> end-to-end on CPU. **Real QLoRA training and real benchmark results are pending
> GPU execution.** Any metrics currently in [reports/benchmark_report.md](reports/benchmark_report.md)
> are **sample / pipeline-validation numbers (mock backend), not model
> performance claims.** See [Limitations](#limitations).

---

## 1. Project overview

FinSage-7B is an end-to-end pipeline that fine-tunes a small open model
(Mistral-7B-Instruct) to analyse U.S. **SEC filings** (10-K / 10-Q / 8-K) —
summarising risk factors, explaining MD&A, extracting reported metrics, and
answering questions **grounded strictly in the filing text**.

- **Why financial filings?** They are long, dense, public, well-structured, and
  high-stakes — an ideal domain to prove that a cheaply fine-tuned model can be
  made *more grounded and specific* than its base model on a real professional
  task, without hallucinating numbers or drifting into advice.
- **What problem it solves.** Analysts spend hours locating risk factors, revenue
  drivers, and reported figures across hundreds of pages. FinSage-7B turns a
  pasted excerpt + a task into a grounded, disclaimer-bounded answer.
- **More than a chatbot.** This is a *system*: leakage-safe data engineering,
  reproducible evaluation, a real serving stack (auth, rate limiting, logging,
  disclaimer injection), a web demo, Docker deployment, and an automated
  benchmark report — all covered by tests and CI.

## 2. Demo status

- ✅ **Demo mode available** — the Next.js frontend runs with no backend and
  returns clearly-labelled mock responses.
- ⚙️ **Full model serving requires a GPU** and a merged FinSage-7B model behind
  the internal vLLM server.
- ⏳ **Real benchmark results pending GPU execution** — see the honesty note above.

## 3. Architecture

**ML pipeline**

```
SEC Filings
  → EDGAR Ingestion
  → Section Extraction
  → Instruction Dataset
  → Baseline Evaluation
  → QLoRA Fine-Tuning
  → Fine-Tuned Evaluation
  → vLLM Serving
  → FastAPI Wrapper
  → Next.js Demo
  → Benchmark Report
```

**Deployment architecture**

```
Browser
  → Next.js Frontend
  → Next.js API Proxy        (injects X-API-Key server-side; secret never in browser)
  → FastAPI Wrapper          (public, CPU: auth, rate limit, logging, disclaimer)
  → vLLM Server              (internal, GPU; never exposed publicly)
  → FinSage-7B Model
```

See [docs/architecture.md](docs/architecture.md) for detail.

## 4. Key features

- SEC EDGAR ingestion (rate-limited, cached, retrying client)
- Filing section extraction (risk factors, MD&A, business, …)
- Instruction dataset generation across 10 task types
- **Leakage-safe** train / validation / test split (by company and time)
- Baseline evaluation of the base model
- QLoRA training scaffold (PEFT + TRL + bitsandbytes)
- Fine-tuned evaluation + base-vs-fine-tuned comparison
- Automated benchmark report (Markdown + PDF + HTML + charts)
- vLLM OpenAI-compatible serving
- FastAPI wrapper: API-key auth, rate limiting, structured logging, disclaimer injection
- Next.js + TypeScript + Tailwind frontend demo
- Docker Compose deployment (demo / full / GPU)
- Mock/demo mode throughout for GPU-free exploration

## 5. Tech stack

| Layer | Tools |
| --- | --- |
| Model | Mistral-7B-Instruct-v0.3 |
| Fine-tuning | QLoRA, PEFT, TRL, bitsandbytes, Accelerate |
| Data | Public SEC EDGAR filings (10-K / 10-Q / 8-K) |
| Serving | vLLM (OpenAI-compatible) |
| Backend | FastAPI, Pydantic v2, Uvicorn |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Evaluation | ROUGE-L, token F1, numeric match, classification accuracy, lexical faithfulness (optional NLI) |
| Reporting | matplotlib, markdown, reportlab |
| Deployment | Docker, Docker Compose |
| Testing | pytest, Vitest |
| Code quality | ruff, black, mypy, TypeScript, GitHub Actions CI |

## 6. Current implementation status

| Phase | Status | Description |
| --- | --- | --- |
| 1. Scaffold | ✅ Done | `src/` package, config, CLIs, optional extras |
| 2. EDGAR ingestion | ✅ Done | Rate-limited client, section extraction, preprocessing |
| 3. Instruction dataset | ✅ Done | 10 task types, leakage-safe splits, validation |
| 4. Baseline evaluation | ✅ Done (pipeline) | Mock backend CPU-only; real base eval needs GPU |
| 5. QLoRA fine-tuning | ⚙️ Implemented; **real training pending GPU run** | Dry-run validated on CPU |
| 6. Fine-tuned eval & benchmark | ✅ Done (pipeline) | Compares base vs fine-tuned; real numbers need a trained adapter |
| 7. vLLM serving | ✅ Done | Client + smoke tests CPU-only; server needs GPU |
| 8. FastAPI wrapper | ✅ Done | Auth, rate limit, logging, disclaimer; 257 tests |
| 9. Next.js frontend | ✅ Done | Demo mode + server-side proxy |
| 10. Docker deployment | ✅ Done | demo / full / GPU compose files |
| 11. Benchmark report | ✅ Done | Markdown + PDF + HTML + charts + validation |
| 12. Portfolio polish | ✅ Done | Docs, publishing assets, final repo check |

> The training **pipeline is implemented and dry-run tested**; the real GPU
> training run is pending. See [docs/roadmap.md](docs/roadmap.md).

## 7. Quickstart

```bash
# Clone
git clone https://github.com/sayujsawant-max/Fine-Tuned-Domain-LLM.git finsage-7b
cd finsage-7b

# Python setup
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# Run the quality gates (lint + typecheck + tests)
make check        # or: ruff check . && black --check . && mypy && pytest

# Frontend demo (no backend needed)
cd frontend && npm install && npm run dev   # http://localhost:3000

# Docker demo (no GPU, no model)
cp .env.example .env
make deploy-demo

# Full GPU stack (NVIDIA Container Toolkit + merged model required)
make deploy-gpu
```

> `make` is unavailable on some Windows setups — run the underlying commands from
> the Makefile directly (e.g. `.venv\Scripts\python.exe -m pytest`).

## 8. Dataset pipeline

- **Public SEC filings only** — no private or proprietary data; no PII.
- **No investment advice** is generated or implied.
- Download filings, extract sections, and build the instruction dataset:

```bash
python scripts/download_edgar.py download --tickers AAPL MSFT --forms 10-K
python scripts/extract_sections.py extract
python scripts/build_instruction_dataset.py build
python scripts/validate_dataset.py validate
```

- **Leakage prevention:** examples are split by company (and optionally time), so
  no company in training appears in the test set. A leakage check is written to
  `dataset_stats.json` / `validation_report.json`. See
  [docs/dataset_guide.md](docs/dataset_guide.md).

## 9. Evaluation

Baseline and fine-tuned models are scored on the **same held-out test set** with
identical prompts and generation settings. Metrics: exact match, token F1,
ROUGE-L, numeric precision/recall/exact-match, classification accuracy, and
lexical faithfulness (optional NLI entailment).

> ⚠️ **Mock metrics validate the evaluation pipeline only. They are not model
> performance claims.** Real numbers require a trained adapter (GPU). See
> [docs/eval_guide.md](docs/eval_guide.md).

## 10. Training

```bash
# Dry-run (CPU): validates data + formatting without loading the model
python training/train.py --dry-run

# Real QLoRA training (GPU; install .[ml,training])
python training/train.py --train-file data/datasets/train.jsonl ...
python training/merge_adapter.py    # merge LoRA adapter into the base model
```

GPU required for real training; the adapter is saved to `checkpoints/finsage-7b`
and can be merged for serving. W&B logging is optional. See
[docs/training_guide.md](docs/training_guide.md).

## 11. Serving and API

- **vLLM** serves the merged model with an OpenAI-compatible API (GPU, internal).
- **FastAPI wrapper** (public, CPU) adds API-key auth (`X-API-Key`), sliding-window
  rate limiting, structured JSON logging (never logs filing text), financial
  disclaimer injection, and `/v1/health` + `/v1/ready` probes.

```bash
bash serving/vllm_server.sh         # GPU host
bash serving/start_api.sh           # public wrapper (no GPU)
curl localhost:8080/v1/health
```

## 12. Frontend demo

Paste a filing excerpt → choose a task → ask a question → view the grounded
response (answer, model, latency, request id, disclaimer). The browser only ever
calls the same-origin Next.js proxy, which injects the API key server-side.
**Demo mode** returns labelled mock responses with no backend running.

## 13. Deployment

```bash
make deploy-demo   # frontend + mock API, no GPU
make deploy-local  # full stack locally
make deploy-gpu    # NVIDIA Container Toolkit required
```

> 🔒 **Security:** never expose the vLLM server publicly — it has no auth. Only the
> FastAPI wrapper should be public. See [reports/deployment_checklist.md](reports/deployment_checklist.md).

## 14. Benchmark report

```bash
make report            # generate from reports/figures
make validate-report   # structural + honesty validation
make report-mock       # clearly-labelled sample report from fixtures
```

Output: `reports/benchmark_report.{md,pdf,html}` + charts. The report
**auto-detects mock data** and stamps a prominent sample/mock banner — mock
numbers are never presented as real. See [docs/eval_guide.md](docs/eval_guide.md).

## 15. Limitations

- **Real QLoRA training requires a GPU**; the run is pending (RunPod/A100).
- **Real benchmark results require an actual trained adapter** — current numbers
  are sample/pipeline-validation only.
- **Weak-supervision labels:** Phase 3 targets are template/extractive (no LLM
  teacher), so references approximate ground truth.
- **Lexical faithfulness is not true NLI** — it is an overlap proxy.
- **API-key auth is simple demo security**, not enterprise SSO/OAuth.
- **The in-memory rate limiter is not multi-replica production-ready** (a Redis
  backend is provided but optional).
- **Not financial advice.** Outputs must be verified against the source filing.

## 16. Roadmap

- Run full QLoRA training on RunPod / A100 and publish real benchmark numbers
- Replace weak-supervision labels with human/LLM-reviewed targets
- Add NLI-based faithfulness scoring and an LLM-as-judge with calibration
- Add a RAG baseline for comparison
- Add a Hugging Face Spaces demo and a public hosted GPU demo
- Add a monitoring dashboard

Full roadmap: [docs/roadmap.md](docs/roadmap.md).

## 17. Repository structure

```
finsage-7b/
├── src/finsage/        # Python package: data, evaluation, training, serving, reporting
├── scripts/            # CLIs: download, extract, build dataset, report, repo checks
├── training/           # QLoRA training entrypoint
├── evaluation/         # baseline / fine-tuned eval CLIs
├── serving/            # vLLM + FastAPI launch scripts
├── frontend/           # Next.js + TypeScript demo
├── docker/             # Compose files (demo / full / GPU)
├── configs/            # YAML configs
├── docs/               # guides, model/dataset cards, interview & publishing guides
├── reports/            # benchmark report, demo script, resume/LinkedIn assets
├── tests/              # pytest suite (CPU-only)
└── .github/            # CI workflow, issue/PR templates
```

## 18. License

Apache-2.0 — see [LICENSE](LICENSE).

## 19. Disclaimer

> **FinSage-7B is not a licensed financial advisor. Outputs are not investment
> recommendations. Always verify responses against the original filing.**

---

**Docs:** [Project summary](docs/project_summary.md) ·
[Interview guide](docs/interview_guide.md) ·
[Reproducibility](docs/reproducibility.md) ·
[Publishing guide](docs/publishing_guide.md) ·
[Model card](docs/model_card.md) · [Dataset card](docs/dataset_card.md) ·
[Roadmap](docs/roadmap.md) · [FAQ](docs/faq.md) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)
