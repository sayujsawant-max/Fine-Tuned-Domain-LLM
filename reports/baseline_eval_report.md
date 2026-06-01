# FinSage-7B — Baseline Evaluation Report

- **Evaluation date:** 2026-06-01
- **Backend:** mock
- **Model ID:** n/a (mock backend)
- **Test file:** tests/fixtures/eval_test_sample.jsonl
- **Examples evaluated:** 10
- **Avg input length (chars):** 132.1
- **Avg prediction length (chars):** 99.7

## Overall metrics

| Metric | Value |
|--------|-------|
| classification_accuracy | 0.5000 |
| exact_match | 0.0000 |
| lexical_faithfulness | 0.8963 |
| numeric_exact_match | 0.0000 |
| numeric_precision | 1.0000 |
| numeric_recall | 0.7500 |
| rouge_l | 0.6220 |
| token_f1 | 0.6208 |

## Metrics by task

### analyst_summary (n=1)

| Metric | Value |
|--------|-------|
| lexical_faithfulness | 1.0000 |
| rouge_l | 0.4000 |
| token_f1 | 0.5333 |

### business_risk_identification (n=1)

| Metric | Value |
|--------|-------|
| lexical_faithfulness | 1.0000 |
| rouge_l | 0.8462 |
| token_f1 | 0.8462 |

### filing_qa (n=1)

| Metric | Value |
|--------|-------|
| exact_match | 0.0000 |
| lexical_faithfulness | 1.0000 |
| token_f1 | 0.8000 |

### hallucination_detection (n=1)

| Metric | Value |
|--------|-------|
| classification_accuracy | 1.0000 |
| lexical_faithfulness | 0.4000 |

### mda_explanation (n=1)

| Metric | Value |
|--------|-------|
| lexical_faithfulness | 1.0000 |
| rouge_l | 0.0690 |
| token_f1 | 0.2069 |

### metric_extraction (n=1)

| Metric | Value |
|--------|-------|
| lexical_faithfulness | 0.6667 |
| numeric_exact_match | 0.0000 |
| numeric_precision | 1.0000 |
| numeric_recall | 0.7500 |
| token_f1 | 0.6667 |

### outlook_classification (n=1)

| Metric | Value |
|--------|-------|
| classification_accuracy | 0.0000 |
| token_f1 | 0.1176 |

### revenue_driver_explanation (n=1)

| Metric | Value |
|--------|-------|
| lexical_faithfulness | 1.0000 |
| rouge_l | 1.0000 |
| token_f1 | 1.0000 |

### risk_summary (n=1)

| Metric | Value |
|--------|-------|
| lexical_faithfulness | 1.0000 |
| rouge_l | 0.4167 |
| token_f1 | 0.4167 |

### yoy_comparison (n=1)

| Metric | Value |
|--------|-------|
| lexical_faithfulness | 1.0000 |
| rouge_l | 1.0000 |
| token_f1 | 1.0000 |

## Task distribution

| Task type | Examples |
|-----------|----------|
| analyst_summary | 1 |
| business_risk_identification | 1 |
| filing_qa | 1 |
| hallucination_detection | 1 |
| mda_explanation | 1 |
| metric_extraction | 1 |
| outlook_classification | 1 |
| revenue_driver_explanation | 1 |
| risk_summary | 1 |
| yoy_comparison | 1 |

## Qualitative examples

- **Task:** risk_summary  
  **Instruction:** Summarize the key risk factors discussed in the filing excerpt.  
  **Reference:** The company discloses competition risk, supply chain disruption risk, and cybersecurity risk.  
  **Prediction:** The company faces competition risk from lower-cost manufacturers. Supply chain disruptions could increase costs.

- **Task:** mda_explanation  
  **Instruction:** Explain the main management discussion points in this MD&A excerpt.  
  **Reference:** Management attributes growth to services demand and improved margins, and plans continued investment.  
  **Prediction:** Revenue grew driven by strong demand in the services segment. Gross margin expanded due to manufacturing efficiencies.

- **Task:** metric_extraction  
  **Instruction:** Extract financial metrics, figures, percentages, or monetary values mentioned in the filing excerpt.  
  **Reference:** - $81,462 million - 51% - 25.6% - $12,556 million  
  **Prediction:** Reported metrics: $81,462, 51%, 25.6%, $12,556.

## Limitations

- This is the **base model before fine-tuning**; results are a reference point, not a target.
- Reference targets are template/extractive weak supervision (Phase 3), so absolute scores should be read with caution.
- `lexical_faithfulness` is a lexical proxy, not a true NLI metric.
- Mock-backend reports exist only to validate the pipeline.

## Disclaimer

> FinSage-7B is not a licensed financial advisor. Outputs are not investment recommendations and may be inaccurate. Always verify against the original filings. This baseline uses the un-fine-tuned base model.
