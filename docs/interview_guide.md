# FinSage-7B — Interview Guide

Talking points and honest answers for discussing this project.

## 30-second explanation

“FinSage-7B is an end-to-end pipeline that fine-tunes Mistral-7B on public SEC
filings to analyse risk factors, MD&A, and reported metrics — grounded in the
filing text. I built the full system: leakage-safe data, evaluation, QLoRA
training, vLLM + FastAPI serving, a Next.js demo, Docker deployment, and an
automated benchmark report. The engineering is complete and tested; the real GPU
training run is the next step.”

## 2-minute explanation

Cover: the problem (filings are long/dense/high-stakes), the data pipeline
(EDGAR ingest → section extraction → instruction dataset with 10 task types and
company/time-based leakage-safe splits), evaluation (same test set, multi-metric,
faithfulness), QLoRA fine-tuning (4-bit base + low-rank adapters for cheap
training), serving (internal vLLM + public FastAPI wrapper with auth/rate-limit/
logging/disclaimer), the demo and Docker deployment, and the automated benchmark
report that honestly labels mock vs real numbers.

## Deep technical explanation

- **Data:** httpx EDGAR client (5 req/s, retries, JSON cache); HTML cleaning +
  Item-heading section detection; instruction builder renders deterministic
  template/extractive targets (flagged weak-supervision).
- **Splits:** partition by company key (and optionally year) so no company leaks
  across train/test; a leakage check is asserted and recorded.
- **Training:** QLoRA = NF4 4-bit quantised base + LoRA adapters via PEFT/TRL;
  forward-compatible with modern TRL (`SFTConfig`, `processing_class`) and
  transformers (`quantization_config`). CPU dry-run validates data + formatting.
- **Eval:** generator abstraction (mock / transformers / adapter / merged), shared
  metrics, base-vs-fine-tuned comparison with per-task deltas.
- **Serving:** vLLM OpenAI-compatible server (GPU, internal); FastAPI wrapper adds
  `X-API-Key` auth (`secrets.compare_digest`), sliding-window rate limiting,
  structured JSON logging (never logs filing text), disclaimer injection, and
  normalised error envelopes.
- **Frontend:** browser → same-origin Next.js proxy (injects the key server-side)
  → FastAPI; demo mode returns labelled mocks.

## Why financial filings?

Public, structured, high-stakes, and genuinely hard — a credible domain to show
grounded, specific answers beat a generic base model, with measurable metrics.

## Why QLoRA?

It makes 7B fine-tuning affordable on a single GPU: 4-bit quantisation slashes
memory while LoRA adapters keep the trainable parameter count tiny, so the run is
cheap and the adapter is small and shareable.

## Why Mistral-7B?

Strong open 7B instruct model with a permissive license and a long context — big
enough to be useful, small enough to fine-tune and serve on modest hardware.

## Why vLLM?

High-throughput, OpenAI-compatible serving with paged-attention KV cache —
production-grade inference without writing a custom server, and a drop-in API the
FastAPI wrapper can proxy.

## Why a FastAPI wrapper (not vLLM directly)?

vLLM has no auth and shouldn’t be public. The wrapper is the security and policy
boundary: authentication, rate limiting, logging, disclaimer injection, and a
clean public contract — while vLLM stays internal.

## How did you avoid data leakage?

Splits are partitioned by company (and optionally time), not random rows, so no
company in training appears in test. The split writes a leakage check; the
dataset validator fails if train/test companies overlap.

## How did you evaluate improvement?

Same held-out test set, identical prompts and generation settings for base and
fine-tuned; multi-metric scoring plus a separate faithfulness track; a comparison
report with per-metric and per-task deltas and selected qualitative examples.

## What are the limitations?

Real training is pending GPU; current numbers are mock/pipeline-validation;
weak-supervision labels; lexical (not NLI) faithfulness by default; demo-grade
auth; in-memory rate limiter; not financial advice.

## What would you improve next?

Run real training and publish real numbers; upgrade labels to human/LLM-reviewed;
add NLI faithfulness + calibrated LLM-as-judge; add a RAG baseline; host a public
GPU demo; add monitoring.

## Questions an interviewer may ask (with strong answers)

- *“Have you trained the model fully yet?”* → **The complete training pipeline is
  implemented and tested in dry-run mode. The real GPU training run is pending
  execution on RunPod/A100. I deliberately separated the engineering pipeline from
  the GPU-dependent run so the repo stays reproducible locally and honest about
  what’s been measured.**
- *“Aren’t the benchmark numbers fake?”* → They’re explicitly labelled
  sample/pipeline-validation from a mock backend; the report auto-detects this and
  banners it. No mock number is presented as real performance.
- *“How is this production-relevant?”* → The serving stack mirrors production:
  internal GPU inference behind a public authenticated CPU wrapper with rate
  limiting, logging, disclaimers, health checks, and Docker deployment.
- *“What was the hardest part?”* → Keeping the whole thing honest and reproducible
  without a GPU — designing mock backends and a benchmark report that validate the
  pipeline without faking results.
