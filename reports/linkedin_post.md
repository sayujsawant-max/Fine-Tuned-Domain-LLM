# FinSage-7B — LinkedIn Post Drafts

## Version A — before real GPU training (use now; no performance claims)

> 🚀 I built **FinSage-7B** — a full end-to-end pipeline for a domain fine-tuned
> LLM that analyses SEC filings (10-K / 10-Q / 8-K).
>
> It’s not a notebook — it’s a system:
> • SEC EDGAR ingestion + section extraction
> • Leakage-safe instruction dataset (10 task types, split by company/time)
> • Baseline + fine-tuned evaluation with an automated benchmark report
> • QLoRA fine-tuning scaffold (Mistral-7B, PEFT/TRL/bitsandbytes)
> • Production-style serving: vLLM behind a FastAPI wrapper (auth, rate limiting,
>   structured logging, disclaimer injection)
> • Next.js + TypeScript demo and Docker Compose deployment
> • Tested CPU-only and wired into GitHub Actions CI
>
> I deliberately separated the engineering pipeline from the GPU-dependent
> training run, so the whole thing is reproducible locally and **honest** about
> what’s been measured — the real QLoRA training run is next (RunPod/A100).
>
> Code + docs 👉 [GitHub link]
>
> #MachineLearning #LLM #FineTuning #MLOps #FastAPI #vLLM #FinTech #AIEngineering

## Version B — after real benchmark (fill placeholders)

> 📈 Update: I fine-tuned **FinSage-7B** (QLoRA on Mistral-7B) on SEC filings and
> ran the full evaluation.
>
> After fine-tuning, FinSage-7B improved **[metric] by [delta]** over the base
> model on a held-out, leakage-safe test set, while serving at **[p95] s P95**
> latency via vLLM behind an authenticated FastAPI wrapper.
>
> Full benchmark report, base-vs-fine-tuned comparison, and a live demo 👉 [link]
>
> _Not financial advice — outputs must be verified against the source filing._
>
> #MachineLearning #LLM #FineTuning #MLOps #FastAPI #vLLM #FinTech #AIEngineering

---

**Reminder:** do not post Version B until real benchmark artifacts exist. Replace
every `[...]` with real numbers from `reports/benchmark_report.md`.
