# FinSage-7B Benchmark Report — Appendix

This appendix defines the metrics, tasks, and methodology referenced by the main
benchmark report.

## Metric Definitions

| Metric | Definition |
| --- | --- |
| Exact match | Fraction of predictions that equal the reference exactly (after normalisation). |
| Token F1 | Token-overlap F1 between prediction and reference. |
| ROUGE-L | Longest-common-subsequence overlap; rewards fluent, ordered overlap. |
| Numeric exact match | Fraction of reference numbers reproduced exactly. |
| Numeric precision / recall | Precision/recall over numeric mentions (for extraction tasks). |
| Classification accuracy | Accuracy on label tasks (outlook, hallucination detection). |
| Lexical faithfulness | Lexical-overlap proxy for how grounded the answer is in the excerpt (default). |
| NLI faithfulness | Optional entailment-based faithfulness (MNLI), off by default. |

## Task Type Definitions

| Task type | What the model must do |
| --- | --- |
| risk_summary | Summarise the key risk factors. |
| mda_explanation | Explain the Management's Discussion & Analysis section. |
| metric_extraction | Extract reported financial metrics/numbers. |
| yoy_comparison | Compare year-over-year figures. |
| business_risk_identification | Identify business-specific risks. |
| revenue_driver_explanation | Explain what drove revenue. |
| filing_qa | Answer a question grounded in the filing. |
| analyst_summary | Produce an analyst-style summary. |
| outlook_classification | Classify the forward-looking outlook. |
| hallucination_detection | Detect unsupported / hallucinated claims. |

## Dataset Split Strategy

Splits are **leakage-safe**: examples are partitioned by company (and optionally
by time) so that no company in the training set appears in the test set. This
prevents the model from memorising company-specific phrasing and inflating
held-out scores. The split and a leakage check are recorded in
`dataset_stats.json` and `validation_report.json`.

## Evaluation Caveats

- Phase 3 targets are **weak supervision** (template/extractive, no LLM teacher),
  so references approximate ground truth.
- The default faithfulness metric is a **lexical proxy**, not a full entailment
  audit; n-gram metrics can penalise correct added detail.
- The held-out set is **small**; treat deltas as directional.
- A mock/sample report uses fabricated, clearly-labelled numbers for pipeline
  validation only.

## Prompt Format

Both models receive the same instruction-style prompt: a task instruction, the
filing excerpt to ground the answer in, and (optionally) a question. The base
model and fine-tuned model use identical generation settings for a fair
comparison.

## Safety Disclaimer

FinSage-7B is **not** a licensed financial advisor. Outputs are informational
summaries of the supplied text only, are **not** investment recommendations, and
may be incomplete or incorrect. Always verify against the original filing.

## Reproducibility Commands

```bash
# Dataset
make build-dataset && make validate-dataset
# Baseline evaluation (mock backend is CPU-only)
make eval-baseline
# Fine-tune (GPU) + evaluate + compare
make train && make eval-finetuned && make compare-models
# Report
make report && make validate-report
```
