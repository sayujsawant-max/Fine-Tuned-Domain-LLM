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
| `lexical_faithfulness` | Share of prediction content words present in the source excerpt (default proxy). |
| `nli_faithfulness` | Entailment probability of the prediction given the source (real NLI; opt-in via `--faithfulness nli`). |

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

## Faithfulness: lexical proxy vs NLI

`lexical_faithfulness` (default) is a **lexical overlap proxy**, not entailment.
It rewards copying source words and cannot detect contradiction, wrong
attribution, or unsupported synthesis — treat it as a rough screen.

For a real entailment signal, pass `--faithfulness nli` to the baseline or
fine-tuned eval CLIs (or `EvalRunner(faithfulness="nli")`). This scores the
prediction as a hypothesis entailed by the source premise using an MNLI model
(`facebook/bart-large-mnli` by default), loaded lazily via the `ml` extra and
cached. The scorer is injectable, so tests stay offline. NLI penalises
contradiction and unsupported claims that lexical overlap misses, at the cost of
a model download and slower scoring.

Phase 3 reference targets are weak supervision (unless upgraded via the optional
Phase 3.5 LLM-assisted pass), so absolute scores should be read as a *reference
point*, not ground truth.

## Future improvements (Phase 6+)

- **NLI faithfulness** (entailment model) replacing the lexical proxy.
- **LLM-as-judge** (disabled by default; objective metrics drive headline claims).
- **RAG baseline** as a third column in the main comparison.
- **Qualitative report** and **before/after charts** in `reports/figures/`.

## Fine-tuned evaluation & comparison (Phase 6)

### Workflow

1. Run the baseline (Phase 4) → `reports/figures/baseline_{results.json,predictions.jsonl}`.
2. Run the fine-tuned model on the **same** test set with `run_finetuned_eval.py`
   → `finetuned_*` files.
3. Compare → `comparison_results.json`, `metric_delta_by_task.json`,
   `comparison_summary.json`, `qualitative_comparisons.jsonl`, charts, and
   `reports/benchmark_report.md`.

`run_finetuned_eval.py` does steps 2–3 in one shot; `compare_models.py` does
step 3 alone from existing outputs.

### Adapter vs merged evaluation

- **adapter** — loads the base model + LoRA adapter via PEFT
  (`--model-id` + `--adapter-path`); 4-bit supported. Cheapest to keep around.
- **merged** — loads a standalone merged model (`--merged-model-path`); simplest
  to serve (Phase 7) and slightly faster at inference.

Both produce identical output schemas, so the comparison code is backend-agnostic.

### Why the same test set is required

The headline result is a **delta**. If the base and fine-tuned models see
different examples, prompts, or metrics, the delta is meaningless. The CLI
re-runs the fine-tuned model over the exact `--test-file` and reuses the Phase 4
metric functions, and the comparison joins predictions **by example id**.

### How comparison metrics are computed

For every overall and per-task metric: `absolute_delta = finetuned − baseline`,
`relative_delta_pct = absolute_delta / |baseline| × 100`, and `improved =
absolute_delta > 0`. Non-finite values are coerced to `0.0`; metrics present on
only one side are compared against `0.0` and recorded in `warnings`.

### How qualitative examples are selected

Predictions are joined by id; `find_best_improvements` / `find_regressions` rank
the joined rows by a metric's per-example delta (default `token_f1`). The report
shows the top side-by-side examples (reference vs base vs fine-tuned).

### Improvement vs regression

An **improvement** is a positive delta on an overall metric; a **regression** is
a negative one. Both are reported — never hide regressions.

### Known limitations

- Phase 3 reference targets are **template/extractive weak supervision**.
- `lexical_faithfulness` is lexical overlap, **not** true NLI faithfulness.
- **Mock mode is not real performance** — it only validates the pipeline.
- Small test sets can mislead; treat single-run deltas with caution.

### Future improvements

- NLI-based faithfulness model; LLM-as-judge with an explicit rubric.
- Human evaluation and citation precision.
- Bootstrap confidence intervals on the deltas.

## Honesty

Report regressions as well as gains. Verify numbers against the raw result files
before they reach the README or model card.

## Phase 11: Benchmark report

The benchmark report (`scripts/generate_benchmark_report.py`) consumes the
artifacts produced by **Phase 4** (`baseline_results.json`,
`baseline_predictions.jsonl`) and **Phase 6** (`finetuned_results.json`,
`comparison_results.json`, `comparison_summary.json`, `metric_delta_by_task.json`,
`qualitative_comparisons.jsonl`), plus dataset stats, the training summary, and
latency benchmarks. Nothing is recomputed — the report only *renders* existing
artifacts, so it is fast, CPU-only, and calls no external services.

**Interpreting metrics.** Overall Results show each metric for the base and
fine-tuned model with absolute and relative deltas; a `▲`/`▼` marker flags
improvement/regression. Task-wise Results break this down per task type.
Faithfulness (lexical by default, optional NLI) is summarised separately because
n-gram metrics can *under*-credit a fine-tuned answer that adds correct,
filing-grounded detail absent from the short reference — the most common cause of
an apparent regression.

**Charts** are generated lazily with matplotlib into
`reports/figures/report_*.png`: overall metrics, per-task mean delta, dataset
distribution, latency percentiles, and faithfulness. Each is skipped (with a
warning) if its data or matplotlib is unavailable; the report never fails on a
missing chart.

**Qualitative examples.** `select_qualitative_examples` picks two best
improvements, one regression (worst single-metric drop), one average case, and one
faithfulness case, truncating long text so filings never bloat the report.

**Mock vs real, and what to rerun.** If the result artifacts carry
`"backend": "mock"` (or core files are missing, or `--mock-mode` is set), the
report is auto-labelled a sample/mock report and **must not** be published as real
results. After a real fine-tune, rerun baseline eval → fine-tuned eval →
`compare-models`, then `make report` to regenerate a publishable report.
