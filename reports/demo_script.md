# FinSage-7B — Demo Script

A script for showing the project live (frontend, GitHub, benchmark report) at
three lengths, plus honest lines for the current pre-GPU-training status. Works in
**demo mode** (no backend) or against the live FastAPI backend.

## 30-second demo

> “This is FinSage-7B — a fine-tuned LLM pipeline for analysing SEC filings. I
> paste a filing excerpt, pick a task like *risk summary*, and it returns an
> answer grounded in that text, with a disclaimer. Behind it is a full stack:
> QLoRA training, vLLM + FastAPI serving, and a Next.js demo — all tested and in
> CI.”

**Show:** the frontend — paste sample → *risk_summary* → Analyze → response panel.

## 2-minute demo

1. **Frontend (≈40s):** paste an excerpt, choose a task, ask a question, show the
   response panel (answer, model, latency, request id, disclaimer). Toggle the
   base-vs-FinSage comparison panel.
2. **Architecture (≈30s):** “Browser → Next.js proxy → FastAPI wrapper → vLLM. The
   API key is injected server-side; vLLM stays internal.”
3. **Benchmark report (≈30s):** open `reports/benchmark_report.md` — overall
   metrics, per-task deltas, charts. Note the sample/mock banner.
4. **Honesty (≈20s):** “Numbers today are pipeline-validation from a mock backend;
   real training on GPU is the next step.”

## 5-minute technical walkthrough

1. **Data:** `scripts/download_edgar.py` → `extract_sections` → instruction
   dataset; show leakage-safe split + `validate_dataset`.
2. **Training:** `training/train.py --dry-run` (CPU) → real QLoRA on GPU →
   `merge_adapter`.
3. **Evaluation:** baseline vs fine-tuned on the same test set; metrics +
   faithfulness; `compare_models`.
4. **Serving:** `serving/vllm_server.sh` (GPU) + `start_api.sh`; hit `/v1/health`,
   show auth/rate-limit/disclaimer.
5. **Report:** `make report` → Markdown + PDF + charts; `make validate-report`.
6. **Quality:** `make check` + `python scripts/final_repo_check.py` + CI.

## What to show on the frontend
Sample filing input, task selector (10 tasks), question box, response panel,
base-vs-FinSage comparison, demo-mode badge.

## What to show in GitHub
README (status table + honesty banner), CI badge green, clean `src/` layout,
tests, docs/ guides, model/dataset cards.

## What to show in the benchmark report
Executive summary, overall metrics table, per-task deltas, charts, qualitative
examples, limitations — and the sample/mock banner.

## What to say if the real model is not yet trained

> “The full training pipeline is implemented and dry-run tested. The real GPU
> training run is pending on RunPod/A100 — I kept the engineering reproducible
> locally and the metrics honest, so nothing here is a fabricated result.”

## What to say if demo mode is used

> “This is demo mode — the responses are clearly-labelled mocks so the UX works
> with no GPU. With the backend attached, the same flow hits the real model
> through the FastAPI wrapper.”

## Ending pitch

> “FinSage-7B shows I can take a domain LLM from raw public data all the way to a
> tested, deployable, honestly-evaluated system — data, training, evaluation,
> serving, frontend, deployment, and reporting. The only thing between this and
> real numbers is a GPU training run, which the pipeline is ready to execute.”
