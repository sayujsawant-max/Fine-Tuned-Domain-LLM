# Architecture

```
SEC EDGAR
   ↓ ingestion (finsage.data.edgar_client)
Section Extraction (finsage.data.section_extractor)
   ↓ cleaning
Chunking (finsage.data.chunker)
   ↓
Instruction Dataset Builder (finsage.data.instruction_builder) → data/datasets/*.jsonl
   ↓
Baseline Evaluation (evaluation/run_baseline_eval.py)
   ↓
QLoRA Fine-Tuning (training/train.py)
   ↓
Fine-Tuned Evaluation (evaluation/run_finetuned_eval.py)
   ↓
Benchmark Report (evaluation/compare_models.py) → reports/benchmark_report.md
   ↓
Adapter / Merged Model (training/merge_adapter.py)
   ↓
vLLM Server (serving/vllm_server.sh)  [internal, OpenAI-compatible]
   ↓
FastAPI Wrapper (finsage.serving.app)  [auth, logging, disclaimer, safety]
   ↓
Frontend Demo (frontend/)
```

## Phase 8 — public API wrapper (current)

```
User / Frontend / API client
        ↓
FastAPI Wrapper (finsage.serving.app, :8080)      ← only public surface
        ├── Auth                 (finsage.serving.auth)
        ├── Rate Limiting        (finsage.serving.rate_limiter / middleware)
        ├── Structured Logging   (finsage.serving.middleware)
        ├── Security Headers     (finsage.serving.middleware)
        ├── Schema Validation    (finsage.serving.schemas)
        ├── Prompt Templating    (finsage.serving.prompt_templates)
        ├── Disclaimer Injection (finsage.serving.disclaimer)
        └── Error Handling       (finsage.serving.errors)
        ↓  VLLMClient (finsage.serving.vllm_client)
vLLM OpenAI-compatible Server (serving/vllm_server.sh, :8000)   ← internal only
        ↓
FinSage-7B merged model (checkpoints/finsage-7b-merged)
```

Routes (`finsage.serving.routes`, prefix `/v1`): `GET /health` (public),
`GET /ready`, `GET /models`, `GET /config`, `POST /chat`,
`POST /chat/completions`. The vLLM endpoint is **internal** (no auth/disclaimer);
the FastAPI wrapper is the only intended-public surface and owns auth, rate
limiting, logging, validation, and disclaimer injection.

## Phase 7 — internal vLLM engine

```
vLLM OpenAI-compatible server → /v1/models (health), /v1/chat/completions (inference)
VLLMClient + health helpers (finsage.serving.vllm_client / health)
Smoke tests (serving/test_endpoint.py) + latency benchmark (serving/benchmark_latency.py)
```

## Components

| Layer | Module / path | Responsibility |
|-------|---------------|----------------|
| Config | `finsage.config` | Env-driven settings (pydantic-settings) |
| Logging | `finsage.logging_utils` | Rich-backed structured logging |
| Data | `finsage.data.*` | EDGAR ingest, extraction, chunking, dataset build |
| Training | `finsage.training.*` | QLoRA trainer, collator, callbacks |
| Evaluation | `finsage.evaluation.*` | Metrics, runner, judge, report |
| Serving | `finsage.serving.*` | FastAPI app, routes, middleware, schemas |

## Boundaries

- **vLLM** is the internal inference engine; it is never exposed publicly.
- **FastAPI** is the public surface and owns auth, logging, prompt templates,
  safety checks, and disclaimer injection.
- Heavy GPU dependencies are isolated in the `training` and `serving` extras so
  the data/eval/API layers run on a plain CPU machine.

> A rendered diagram (`reports/figures/architecture.png`) is added in Phase 12.
