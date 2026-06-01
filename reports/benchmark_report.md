# FinSage-7B Benchmark Report

> ⚠️ **Sample/mock numbers are for pipeline validation only and are not real benchmark results.** Run the adapter/merged backend on a GPU for real figures.

## Executive summary

FinSage-7B (fine-tuned) improved **5/8** overall metrics versus the base model, with a mean absolute delta of **+0.1764** across metrics.

## Models

- **Base model:** mistralai/Mistral-7B-Instruct-v0.3
- **Fine-tuned backend:** mock
- **Fine-tuned model/adapter:** see run config

## Dataset

- **Test file:** tests/fixtures/eval_test_sample.jsonl
- **Examples:** 10 (same held-out set as the baseline)

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

## Overall metrics (base vs fine-tuned)

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| classification_accuracy | 0.5000 | 0.5000 | +0.0000 | +0.0% | ➖ |
| exact_match | 0.1000 | 0.0000 | -0.1000 | -100.0% | 🔻 |
| lexical_faithfulness | 0.7000 | 0.8963 | +0.1963 | +28.0% | ✅ |
| numeric_exact_match | 0.2500 | 0.0000 | -0.2500 | -100.0% | 🔻 |
| numeric_precision | 0.4000 | 1.0000 | +0.6000 | +150.0% | ✅ |
| numeric_recall | 0.3500 | 0.7500 | +0.4000 | +114.3% | ✅ |
| rouge_l | 0.3000 | 0.6449 | +0.3449 | +115.0% | ✅ |
| token_f1 | 0.4200 | 0.6398 | +0.2198 | +52.3% | ✅ |

## Metrics by task

### analyst_summary

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| lexical_faithfulness | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |
| rouge_l | 0.0000 | 0.4000 | +0.4000 | +100.0% | ✅ |
| token_f1 | 0.0000 | 0.5333 | +0.5333 | +100.0% | ✅ |

### business_risk_identification

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| lexical_faithfulness | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |
| rouge_l | 0.0000 | 0.8462 | +0.8462 | +100.0% | ✅ |
| token_f1 | 0.0000 | 0.8462 | +0.8462 | +100.0% | ✅ |

### filing_qa

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| exact_match | 0.0000 | 0.0000 | +0.0000 | +0.0% | ➖ |
| lexical_faithfulness | 0.7200 | 1.0000 | +0.2800 | +38.9% | ✅ |
| token_f1 | 0.4000 | 0.8000 | +0.4000 | +100.0% | ✅ |

### hallucination_detection

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| classification_accuracy | 0.5000 | 1.0000 | +0.5000 | +100.0% | ✅ |
| lexical_faithfulness | 0.6000 | 0.4000 | -0.2000 | -33.3% | 🔻 |

### mda_explanation

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| lexical_faithfulness | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |
| rouge_l | 0.0000 | 0.1714 | +0.1714 | +100.0% | ✅ |
| token_f1 | 0.0000 | 0.3429 | +0.3429 | +100.0% | ✅ |

### metric_extraction

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| lexical_faithfulness | 0.6600 | 0.6667 | +0.0067 | +1.0% | ✅ |
| numeric_exact_match | 0.2500 | 0.0000 | -0.2500 | -100.0% | 🔻 |
| numeric_precision | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |
| numeric_recall | 0.0000 | 0.7500 | +0.7500 | +100.0% | ✅ |
| token_f1 | 0.5000 | 0.6667 | +0.1667 | +33.3% | ✅ |

### outlook_classification

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| classification_accuracy | 0.5000 | 0.0000 | -0.5000 | -100.0% | 🔻 |
| token_f1 | 0.3000 | 0.1176 | -0.1824 | -60.8% | 🔻 |

### revenue_driver_explanation

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| lexical_faithfulness | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |
| rouge_l | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |
| token_f1 | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |

### risk_summary

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| lexical_faithfulness | 0.7400 | 1.0000 | +0.2600 | +35.1% | ✅ |
| rouge_l | 0.3100 | 0.4516 | +0.1416 | +45.7% | ✅ |
| token_f1 | 0.4500 | 0.4516 | +0.0016 | +0.4% | ✅ |

### yoy_comparison

| Metric | Base | FinSage-7B | Δ abs | Δ % | Improved |
|--------|------|------------|-------|-----|----------|
| lexical_faithfulness | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |
| rouge_l | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |
| token_f1 | 0.0000 | 1.0000 | +1.0000 | +100.0% | ✅ |

## Best improvements

- **numeric_precision**: 0.4000 → 1.0000 (+0.6000)
- **numeric_recall**: 0.3500 → 0.7500 (+0.4000)
- **rouge_l**: 0.3000 → 0.6449 (+0.3449)
- **token_f1**: 0.4200 → 0.6398 (+0.2198)
- **lexical_faithfulness**: 0.7000 → 0.8963 (+0.1963)

## Regressions / failure cases

- **numeric_exact_match**: 0.2500 → 0.0000 (-0.2500)
- **exact_match**: 0.1000 → 0.0000 (-0.1000)

## Qualitative examples

**Task:** risk_summary — ACME-2022-10-K-risk_factors-0-risk_summary  
**Instruction:** Summarize the key risk factors.  
**Reference:** Key risks: competition, supply chain, cybersecurity.  
**Base model:** The company faces competition risk.  
**FinSage-7B:** The company faces competition risk from lower-cost manufacturers. Supply chain disruptions could increase costs. Cybersecurity incidents represent a growing risk to operations.  
**Metric change:** {'lexical_faithfulness': 0.3, 'rouge_l': 0.1516, 'token_f1': 0.0516}

**Task:** mda_explanation — ACME-2022-10-K-mda-0-mda_explanation  
**Instruction:** Explain the MD&A.  
**Reference:** Management cites demand growth and margin expansion.  
**Base model:** Revenue grew.  
**FinSage-7B:** Revenue grew driven by strong demand in the services segment. Gross margin expanded due to manufacturing efficiencies. Management expects continued investment next year.  
**Metric change:** {'lexical_faithfulness': 0.32, 'rouge_l': -0.1086, 'token_f1': -0.0071}

**Task:** metric_extraction — ACME-2022-10-K-financial_statements-0-metric_extraction  
**Instruction:** Extract metrics.  
**Reference:** - $81,462 million - 51%  
**Base model:** Reported metrics: $81,462 million.  
**FinSage-7B:** Reported metrics: $81,462, 51%, 25.6%, $12,556.  
**Metric change:** {'lexical_faithfulness': 0.0067, 'numeric_exact_match': 0.0, 'numeric_precision': 1.0, 'numeric_recall': 0.75, 'token_f1': 0.1667}

**Task:** yoy_comparison — ACME-2022-10-K-mda-1-yoy_comparison  
**Instruction:** Identify YoY comparison.  
**Reference:** - Services revenue increased 14% year over year.  
**Base model:** Services revenue increased.  
**FinSage-7B:** Services revenue increased 14% year over year while product revenue grew 6% compared with the prior year.  
**Metric change:** {'lexical_faithfulness': 0.25, 'rouge_l': 0.6, 'token_f1': 0.5}

**Task:** business_risk_identification — ACME-2022-10-K-business-0-business_risk_identification  
**Instruction:** Identify business risks.  
**Reference:** - The business depends on a few suppliers and faces competition.  
**Base model:** The business depends on a few suppliers.  
**FinSage-7B:** The business depends on a limited number of suppliers and faces intense competition. Regulatory changes introduce uncertainty.  
**Metric change:** {'lexical_faithfulness': 0.2, 'rouge_l': 0.3962, 'token_f1': 0.2962}

## Faithfulness / hallucination

Faithfulness is currently approximated by `lexical_faithfulness` (lexical overlap with the source). A true NLI-based faithfulness metric and citation precision are planned (see eval guide).

## Limitations

- Phase 3 reference targets are template/extractive weak supervision.
- `lexical_faithfulness` is a proxy, not entailment.
- Small test sets and mock backends can mislead — read deltas with caution.

## Disclaimer

> FinSage-7B is not a licensed financial advisor. Outputs are not investment recommendations and may be inaccurate. Always verify against the original filings. This baseline uses the un-fine-tuned base model.

## Next steps

- Run the real adapter/merged backend on a GPU for headline numbers.
- Add NLI faithfulness, an LLM-as-judge rubric, and bootstrap confidence intervals.
- Serve the merged model (Phase 7, vLLM).
