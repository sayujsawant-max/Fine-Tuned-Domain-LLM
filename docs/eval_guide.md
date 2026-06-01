# Evaluation Guide (Phases 5 & 7)

The before/after benchmark is the core contribution. We compare **base
Mistral-7B vs FinSage-7B** on a held-out, leakage-free test split.

## Install

```bash
make install-ml   # transformers, datasets, evaluate, rouge-score, bert-score, sklearn
```

## Run

```bash
make eval-baseline    # base model  -> reports/base_results.json
make eval-finetuned   # FinSage-7B  -> reports/finetuned_results.json
make report           # -> reports/benchmark_report.md
```

## Metrics

| Task | Primary | Secondary |
|------|---------|-----------|
| Filing QA | Exact Match | Token F1 |
| Risk Summary | ROUGE-L | BERTScore |
| Metric Extraction | Numeric Exact Match | Unit accuracy |
| Outlook Classification | Accuracy | Macro F1 |
| Hallucination | Faithfulness (NLI) | Citation precision |

Phase 1 ships dependency-free metric implementations in
`finsage.evaluation.metrics` (exact match, token F1, an LCS-based ROUGE-L
placeholder). Phase 7 swaps in real ROUGE/BERTScore behind the `ml` extras.

## Optional baselines

- **RAG baseline:** answer the same questions with retrieval over the filings —
  the strongest framing puts this in the main table, not an appendix.
- **LLM judge:** disabled by default in
  [../configs/eval_config.yaml](../configs/eval_config.yaml); enable for
  faithfulness scoring only, and prefer objective metrics for headline claims.

## Honesty

Report regressions as well as gains. Verify numbers against the raw result files
before they reach the README or model card.
