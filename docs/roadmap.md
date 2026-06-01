# FinSage-7B — Roadmap

## Completed

- ✅ Project scaffold, config, CLIs, optional dependency groups
- ✅ SEC EDGAR ingestion + section extraction + preprocessing
- ✅ Leakage-safe instruction dataset (10 task types) + validation
- ✅ Baseline evaluation pipeline (mock + transformers backends)
- ✅ QLoRA training pipeline (CPU dry-run validated)
- ✅ Fine-tuned evaluation + base-vs-fine-tuned comparison
- ✅ vLLM OpenAI-compatible serving (client + smoke tests)
- ✅ FastAPI wrapper (auth, rate limit, logging, disclaimer)
- ✅ Next.js frontend demo (demo mode + server-side proxy)
- ✅ Docker Compose deployment (demo / full / GPU)
- ✅ Benchmark report generation (Markdown + PDF + HTML + charts)
- ✅ Portfolio polish (docs, publishing assets, final repo check, CI)

## Pending GPU execution

- ⏳ Run full QLoRA training on RunPod / A100
- ⏳ Run real base + fine-tuned evaluation on the held-out test set
- ⏳ Publish **real** benchmark numbers (replace sample/mock report)
- ⏳ Merge + upload the adapter to Hugging Face

## Research upgrades

- Replace weak-supervision labels with human/LLM-reviewed targets
- Add NLI-based faithfulness scoring (entailment)
- Add a calibrated LLM-as-judge with an explicit rubric
- Add a RAG baseline for comparison
- Bootstrap confidence intervals on metric deltas

## Production upgrades

- Redis-backed distributed rate limiting (already pluggable) as default for multi-replica
- OAuth / API-gateway auth instead of a shared key
- Observability: request tracing, latency/error dashboards, alerting
- Autoscaling vLLM workers; canary + rollback for model updates

## Demo upgrades

- Hugging Face Spaces demo (frontend in demo mode)
- Public hosted GPU demo (FastAPI public, vLLM internal)
- Recorded walkthrough video and side-by-side base-vs-fine-tuned comparison
