# FinSage-7B — Project Summary

## One-line summary

An end-to-end, reproducible pipeline that fine-tunes Mistral-7B on public SEC
filings (QLoRA) and serves it through a production-style vLLM + FastAPI + Next.js
stack — with leakage-safe data, automated evaluation, and an honest benchmark
report.

## Problem

Reading SEC filings (10-K / 10-Q / 8-K) is slow, repetitive, and error-prone.
Analysts hunt across hundreds of pages for risk factors, MD&A drivers, and
reported metrics. General-purpose chatbots hallucinate numbers and drift into
advice. The task needs a model that stays **grounded in the provided filing
text** and is bounded by a clear disclaimer.

## Solution

A domain-specialised LLM pipeline:

1. **Ingest** public filings from SEC EDGAR (rate-limited, cached).
2. **Extract** the relevant sections (risk factors, MD&A, business, …).
3. **Build** a leakage-safe instruction dataset across 10 task types.
4. **Evaluate** the base model to establish a baseline.
5. **Fine-tune** Mistral-7B with QLoRA (PEFT + TRL + bitsandbytes).
6. **Re-evaluate** and compare base vs fine-tuned on the same test set.
7. **Serve** the merged model via vLLM behind a public FastAPI wrapper.
8. **Demo** it in a Next.js app; **deploy** via Docker Compose.
9. **Report** the comparison automatically (Markdown + PDF + charts).

## Why this project is impressive

- It is a **system**, not a notebook: data engineering, evaluation, serving,
  frontend, deployment, and reporting — all tested and wired into CI.
- **Honesty by construction:** mock/sample data is auto-detected and labelled; no
  fabricated metrics. This is exactly the discipline ML teams want.
- **Production-style serving:** auth, rate limiting, structured logging,
  disclaimer injection, health probes, and a security boundary (internal vLLM,
  public wrapper).
- **Leakage-safe evaluation:** company/time-based splits prevent inflated scores.
- **Reproducible on commodity hardware:** everything but training runs CPU-only.

## System architecture

```
Browser → Next.js Frontend → Next.js API Proxy → FastAPI Wrapper → vLLM → FinSage-7B
```

The browser never sees the API key; the proxy injects it server-side. vLLM is
internal-only (no auth); only the FastAPI wrapper is public.

## ML pipeline

```
EDGAR → sections → instruction dataset → baseline eval → QLoRA → fine-tuned eval → benchmark report
```

## Evaluation approach

Base and fine-tuned models are scored on the **same** held-out test set with
identical prompts. Metrics: exact match, token F1, ROUGE-L, numeric
precision/recall/exact-match, classification accuracy, and lexical faithfulness
(optional NLI). Faithfulness is reported separately because n-gram metrics can
under-credit a fine-tuned answer that adds correct, filing-grounded detail.

## Deployment architecture

Docker Compose orchestrates three tiers — `frontend` (CPU) → `api` (public CPU
FastAPI) → `vllm` (internal GPU) — with demo, full, and GPU overlays.

## Current status

Phases 1–12 are implemented and tested on CPU. **Real QLoRA training and real
benchmark results are pending GPU execution**; current report numbers are
sample/pipeline-validation only.

## Limitations

GPU-dependent training not yet run; weak-supervision labels; lexical (not NLI)
faithfulness by default; demo-grade auth; in-memory rate limiter; not financial
advice. See the README Limitations section.

## Future work

Real training on RunPod/A100, human/LLM-reviewed targets, NLI + LLM-as-judge,
RAG baseline, Hugging Face Spaces demo, monitoring. See
[roadmap.md](roadmap.md).
