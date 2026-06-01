# Evaluation Guide

The before/after benchmark is the core contribution: **base Mistral-7B vs
FinSage-7B** on a held-out, leakage-free test split. Phase 4 implements the
**baseline** half (the base model); Phase 7 reuses the same harness for the
fine-tuned model and the comparison.

## Goals

- Establish reproducible "before" numbers for the base model.
- Use identical prompts, test data, and metrics for base and fine-tuned runs so
  the delta is meaningful.
- Keep the pipeline CPU-runnable (mock backend) for tests/CI, with an optional
  real GPU backend.

## Backends

- **`mock`** — deterministic, dependency-free generator
  (`finsage.evaluation.generators.MockGenerator`). Used by tests and to validate
  the pipeline end to end. **Its scores are plumbing checks, not a real model
  baseline.**
- **`transformers`** — real Hugging Face inference
  (`TransformersGenerator`); imports torch/transformers lazily and supports the
  tokenizer chat template, 4-bit loading, and greedy decoding. Requires
  `pip install -e ".[ml,training]"` and is GPU-bound for Mistral-7B.

## Run

```bash
# Mock (CPU, no downloads)
make eval-baseline

# Real base model (GPU)
make eval-baseline-real
# or:
python evaluation/run_baseline_eval.py --test-file data/datasets/test.jsonl \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 --backend transformers \
  --device auto --load-in-4bit --max-examples 200
```

## Test set requirements

JSONL with `id`, `instruction`, `input`, `output`, `task_type`, `source`,
`metadata` (the Phase 3 dataset format). The held-out `test.jsonl` must come
from companies not seen in training (company-holdout split).

## Metric definitions

| Metric | Meaning |
|--------|---------|
| `exact_match` | Normalised string equality (SQuAD-style). |
| `token_f1` | Token-overlap F1 after normalisation. |
| `rouge_l` | LCS-based F-measure (hand-rolled; no `rouge-score` dependency). |
| `numeric_exact_match` / `numeric_precision` / `numeric_recall` | Agreement on extracted `$`/`%`/magnitude/number values. |
| `classification_accuracy` | Label match for outlook (pos/neutral/neg) and hallucination (supported/unsupported). |
| `lexical_faithfulness` | Share of prediction content words present in the source excerpt. |

### Task-specific metrics

| Task type | Metrics |
|-----------|---------|
| filing_qa | exact_match, token_f1, lexical_faithfulness |
| risk_summary / mda_explanation / yoy_comparison / business_risk_identification / revenue_driver_explanation / analyst_summary | rouge_l, token_f1, lexical_faithfulness |
| metric_extraction | numeric_match, token_f1, lexical_faithfulness |
| outlook_classification | classification_accuracy, token_f1 |
| hallucination_detection | classification_accuracy, lexical_faithfulness |

## Output files

- `reports/figures/baseline_predictions.jsonl` — per-example id, task, reference, prediction, metrics.
- `reports/figures/baseline_results.json` — overall + per-task aggregates and run metadata.
- `reports/figures/baseline_metrics_by_task.json` — per-task metric breakdown.
- `reports/baseline_eval_report.md` — human-readable Markdown report.

## Known limitations of lexical faithfulness

`lexical_faithfulness` is a **lexical overlap proxy**, not entailment. It rewards
copying source words and cannot detect contradiction, wrong attribution, or
unsupported synthesis. Treat it as a rough screen, not a faithfulness guarantee.
Phase 3 reference targets are also weak supervision, so absolute baseline scores
should be read as a *reference point*, not ground truth.

## Future improvements (Phase 6+)

- **NLI faithfulness** (entailment model) replacing the lexical proxy.
- **LLM-as-judge** (disabled by default; objective metrics drive headline claims).
- **RAG baseline** as a third column in the main comparison.
- **Qualitative report** and **before/after charts** in `reports/figures/`.

## Honesty

Report regressions as well as gains. Verify numbers against the raw result files
before they reach the README or model card.
