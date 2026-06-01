# FinSage-7B — Demo Script

A short, repeatable script for showing the FinSage-7B demo to recruiters and in
interviews. Works in **demo mode** (no backend) or against the live FastAPI
backend.

> Start the UI: `cd frontend && npm run dev` → open <http://localhost:3000>.
> Demo mode needs no backend; for live mode start vLLM + the API first (Phase 8).

---

## 30-second recruiter demo

1. **Open the demo.** One sentence: *"FinSage-7B is a Mistral-7B model fine-tuned
   on SEC filings to answer filing questions without hallucinating."*
2. Click **Load sample → Risk Factors** (auto-fills the excerpt and task).
3. Click **Analyze Filing.**
4. Point at the result: *"Grounded answer, the model name, latency, a request ID
   for tracing, and a mandatory disclaimer — it never gives investment advice."*
5. Tick **Compare vs base Mistral**: *"The base model is vague and drifts into
   advice; the fine-tuned model is specific and stays grounded in the text."*

## 2-minute technical demo

1. **Architecture flow** (bottom of the page): *"The browser calls a server-side
   Next.js proxy, which injects the API key and forwards to a FastAPI wrapper
   that owns auth, rate limiting, logging, and disclaimer injection. Behind it,
   vLLM serves the merged 7B model. The browser never sees the API key."*
2. **Task types:** switch the selector — *"Ten filing-analysis tasks: risk
   summary, MD&A explanation, metric extraction, YoY comparison, and more. Each
   has a suggested question."*
3. **Edit the question** and re-run to show the response panel update with new
   latency / request ID.
4. **Benchmark summary:** *"Sample numbers here; the real before/after gains come
   from the Phase 6 evaluation harness (ROUGE-L, token-F1, faithfulness)."*
5. **Demo vs live mode:** *"With no backend it returns a clearly-labelled mock so
   the UI is always explorable; with the backend up it shows real output and real
   errors (401/429/503)."*

## Interview talking points

**On fine-tuning:** *"I adapted Mistral-7B-Instruct with QLoRA (4-bit) on an
instruction dataset built from public SEC filings — 10 task types, leakage-safe
company/time splits. QLoRA let me train a 7B model on a single GPU."*

**On evaluation:** *"I measured before/after on a held-out set: exact match,
token-F1, ROUGE-L, numeric accuracy, classification, and a lexical-faithfulness
metric, with an optional LLM judge. The benchmark report quantifies the lift over
the base model rather than just claiming it."*

**On deployment:** *"Serving is layered: vLLM gives an OpenAI-compatible,
high-throughput backend (internal only); a FastAPI wrapper adds auth, rate
limiting, structured logging, request IDs, and disclaimer injection; the Next.js
frontend talks to a server-side proxy so the API key stays private. Everything is
containerized with Docker Compose, and the whole CPU path is tested without a
GPU."*

**On safety / honesty:** *"Every answer carries a disclaimer and is grounded in
the provided excerpt; the system prompt forbids investment advice and instructs
the model to say when the excerpt doesn't support an answer. The demo uses only
fabricated sample text — no copyrighted filings."*

## Reset between demos

- Click **Clear** on the filing input, or reload the page.
- Switch tasks to refresh the suggested question.
