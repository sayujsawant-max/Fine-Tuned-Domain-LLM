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
