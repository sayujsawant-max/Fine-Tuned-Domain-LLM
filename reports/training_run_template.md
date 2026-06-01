# FinSage-7B — Training Run Report (template)

> Copy this file per run (e.g. `reports/training_run_YYYYMMDD.md`) and fill in.

## Run metadata

- **Run date:** _YYYY-MM-DD_
- **Model ID:** mistralai/Mistral-7B-Instruct-v0.3
- **Dataset version / commit:** _git SHA of data_
- **Train examples:** _n_
- **Validation examples:** _n_
- **Hardware:** _e.g. 1× A100 80GB (RunPod) / RTX 4090 24GB_

## LoRA config

- r / alpha / dropout: 16 / 32 / 0.05
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- bias: none · task_type: CAUSAL_LM

## Quantization config

- 4-bit NF4, double quant, compute dtype bfloat16

## Hyperparameters

- epochs / lr / scheduler: 3 / 2e-4 / cosine (warmup 0.03)
- batch size × grad-accum: 2 × 8
- max_grad_norm: 0.3 · max_seq_length: 2048 · packing: true
- precision: bf16

## Tracking

- **W&B run:** _link_

## Observations

- **Training loss:** _trend, final value_
- **Eval loss:** _trend, final value, best checkpoint step_
- **Issues:** _OOM, NaN loss, instability, data problems_

## Artifacts

- Adapter: `checkpoints/finsage-7b/`
- `training_summary.json`: _final_train_loss / final_eval_loss_

## Next phase

Phase 6 — evaluate the fine-tuned model with the Phase 4 harness and compare to
the base-model baseline (before/after benchmark).
