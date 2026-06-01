# Model Card — FinSage-7B

> ⚠️ **Model card template. Real model weights and evaluation numbers are pending
> GPU training.** Sections marked _[pending real training]_ must be filled with
> real artifacts before publishing; do not present sample/mock numbers as real.

## Model name

FinSage-7B — a QLoRA fine-tune of Mistral-7B-Instruct for SEC filing analysis.

## Model description

A domain-specialised instruction model that analyses U.S. SEC filings (10-K /
10-Q / 8-K): risk summarisation, MD&A explanation, metric extraction, filing QA,
and related tasks — designed to stay grounded in the supplied filing text.

## Base model

`mistralai/Mistral-7B-Instruct-v0.3`.

## Fine-tuning method

QLoRA — 4-bit (NF4) quantised base + LoRA adapters (PEFT) trained with TRL’s
SFT trainer. Adapters are merged into the base model for serving. _[pending real
training: report final hyperparameters, steps, and hardware]_

## Training data

A leakage-safe instruction dataset built from **public** SEC EDGAR filings across
10 task types, split by company (and optionally time). See
[dataset_card.md](dataset_card.md). Targets are weak-supervision
(template/extractive), not human-verified.

## Intended use

- Assisting analysts/researchers in summarising and querying filing excerpts.
- Educational and portfolio demonstration of domain LLM fine-tuning.

## Out-of-scope use

- **Investment decisions or financial advice.**
- Any high-stakes decision without human verification against the source filing.
- Non-SEC or non-English filings (untested).

## How to use

```python
# After real training + merge, served via vLLM (OpenAI-compatible) behind the
# FinSage FastAPI wrapper:
curl -X POST localhost:8080/v1/chat \
  -H "X-API-Key: $API_SECRET_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key risks.", "filing_excerpt": "...", "task_type": "risk_summary"}'
```

## Evaluation

Base vs fine-tuned on the same held-out test set: exact match, token F1, ROUGE-L,
numeric precision/recall/exact-match, classification accuracy, lexical
faithfulness (optional NLI). _[pending real training: insert real metric table
from `reports/benchmark_report.md`]_

## Limitations

GPU-dependent training; weak-supervision labels; lexical (not NLI) faithfulness by
default; small evaluation set; single base model; no live market data.

## Ethical considerations

Outputs may be incomplete or wrong and must not drive financial decisions. A
financial disclaimer is injected at serving time. No private data is used.

## Financial disclaimer

**FinSage-7B is not a licensed financial advisor. Outputs are not investment
recommendations. Always verify responses against the original filing.**

## Citation / attribution

Built on Mistral-7B-Instruct (Apache-2.0) and public SEC EDGAR data. Project:
`github.com/sayujsawant-max/Fine-Tuned-Domain-LLM`.

## Training status

**Pipeline implemented and dry-run tested; real GPU training pending.** No real
adapter weights are published yet.
