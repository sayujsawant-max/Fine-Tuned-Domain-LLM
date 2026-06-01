# FinSage-7B — Resume Bullets

Pick the set that matches reality. **Use Set A until real GPU training has run;**
only switch to Set B (with real numbers) after a real benchmark exists.

## Set A — before real GPU training (use now)

- Built an **end-to-end LLM fine-tuning pipeline** for SEC filing analysis —
  EDGAR ingestion, instruction-dataset generation, baseline evaluation, a QLoRA
  training scaffold, vLLM serving, a FastAPI API layer, a Next.js demo, and Docker
  deployment — all tested (CPU-only) and wired into GitHub Actions CI.
- Implemented **leakage-safe dataset splitting** (by company/time), multi-task
  financial-NLP **evaluation** (ROUGE-L, token F1, numeric match, classification
  accuracy, faithfulness), and **automated benchmark report generation** comparing
  base vs fine-tuned Mistral-7B.
- Designed a **production-style LLM serving stack**: vLLM behind a public FastAPI
  wrapper with API-key auth, sliding-window rate limiting, structured logging,
  financial-disclaimer injection, health probes, and Docker Compose deployment.
- Engineered the project for **reproducibility and honesty**: mock/demo backends
  for GPU-free validation and a report that auto-labels sample vs real results, so
  no unverified metric is presented as real.

## Set B — after real GPU training (fill placeholders)

> Replace every `[...]` with real values from `reports/benchmark_report.md`. Do
> not use these until a real adapter has been trained and evaluated.

- Fine-tuned **Mistral-7B on [N] SEC-filing instruction examples using QLoRA**,
  improving **[metric] from [base] to [fine-tuned]** on a held-out, leakage-safe
  test set.
- Deployed the merged **FinSage-7B** model with **vLLM**, achieving **[p95] s P95
  latency** at **[throughput] tokens/s** behind an authenticated FastAPI wrapper.
- Reduced hallucination via fine-tuning, raising **faithfulness from [base] to
  [fine-tuned]** while keeping answers grounded in the source filing.

## One-liner (LinkedIn headline / resume summary)

> Built FinSage-7B: an end-to-end QLoRA fine-tuning + vLLM/FastAPI/Next.js serving
> pipeline for SEC-filing analysis, with leakage-safe data, automated evaluation,
> and Docker deployment.
