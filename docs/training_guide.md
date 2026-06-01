# Training Guide (Phase 5)

## QLoRA overview

QLoRA fine-tunes a large model cheaply: the base model is **frozen** and loaded
in **4-bit NF4**, and gradients flow only through a small **LoRA adapter** added
to the attention + MLP projections. This cuts memory enough to fine-tune
Mistral-7B on a single consumer GPU, while the trained adapter stays a few MB.

## Hardware requirements

- **GPU required.** A single 24 GB GPU (RTX 3090/4090) works; A100 80 GB is
  faster. `bitsandbytes` 4-bit needs CUDA — native Windows is painful, prefer
  Linux, WSL2, or a cloud GPU (RunPod/Colab).
- The CPU **dry-run** path needs none of this.

## Install

```bash
pip install -e ".[ml,training]"   # torch, transformers, peft, trl, bitsandbytes, accelerate, wandb, datasets
```

## Dataset requirements

Train/validation JSONL from Phase 3, each example carrying `instruction`,
`input`, `output`, `task_type` (validated before any model loads). Splits must
be company-holdout (no train/test company overlap).

## Prompt formatting

Each example is rendered by `finsage.training.data_formatter.format_sft_example`
into a single `text` field (Mistral instruct format) that embeds a disclaimer
instruction and **never** provides investment advice:

```
<s>[INST] You are FinSage-7B, a financial filing analysis assistant.
Answer using only the provided filing excerpt. Do not provide investment advice.

Task Type: ...
Instruction: ...
Filing Excerpt: ...
[/INST]
{output}</s>
```

## LoRA configuration ([configs/lora_config.yaml](../configs/lora_config.yaml))

- `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `bias=none`, `task_type=CAUSAL_LM`.
- `target_modules`: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  (attention + MLP projections — the standard high-coverage set for Mistral).

## Quantization configuration

4-bit **NF4** with double quantization and bfloat16 compute dtype
(`BitsAndBytesConfig`). This is the "Q" in QLoRA.

## Training hyperparameters ([configs/training_config.yaml](../configs/training_config.yaml))

3 epochs · lr 2e-4 cosine (warmup 0.03) · batch 2 × grad-accum 8 ·
`max_grad_norm=0.3` · gradient checkpointing · bf16 · `max_seq_length=2048` ·
packing · `save_steps/eval_steps=200` · `load_best_model_at_end` on `eval_loss`.

## Library-version compatibility

The trainer is written to work across library generations without code changes:

- **TRL** — `build_training_args`/`build_sft_trainer` prefer `SFTConfig` and
  introspect the `SFTTrainer` signature, so they use `processing_class` (modern
  TRL ≥0.12) or `tokenizer` (legacy) and place the SFT fields wherever the
  installed version expects them.
- **transformers** — 4-bit loading uses `quantization_config=BitsAndBytesConfig`
  (the `load_in_4bit=True` `from_pretrained` kwarg was removed in recent
  releases).

If a future TRL release renames a field the introspection doesn't catch, pin the
working versions from the `training` extra and report it.

## Common failure points

- **NaN/Inf loss** — `build_nan_loss_callback` stops training; lower the learning
  rate, check for degenerate examples, ensure bf16 (not fp16) on Ampere+.
- **OOM** — reduce `per_device_train_batch_size`, increase
  `gradient_accumulation_steps`, lower `max_seq_length`, keep gradient
  checkpointing on, ensure 4-bit loading is active.
- **Tokenizer has no pad token** — handled (pad := eos, right padding).

## Resume training

```bash
python training/train.py ... --resume-from-checkpoint checkpoints/finsage-7b/checkpoint-1200
```

## Adapter saving & merging

`train()` saves the adapter, tokenizer, and `training_summary.json` to
`--output-dir`. For deployment, merge the adapter into the base weights:

```bash
python training/merge_adapter.py --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter-path checkpoints/finsage-7b --output-dir checkpoints/finsage-7b-merged
```

## W&B tracking

`--report-to wandb` (default) logs loss curves; run `wandb login` first, or use
`--report-to none` for an offline run. `checkpoints/` and `wandb/` are git-ignored.

## Dry-run

`python training/train.py ... --dry-run` (or `make train-dry-run`) validates
files, configs, and dataset schema and previews formatting **without importing
torch/transformers or loading a model** — the fast pre-flight check before
spending GPU time.

## Next: Phase 6 — fine-tuned evaluation

Phase 6 reuses the Phase 4 harness (`EvalRunner` + `TransformersGenerator` loaded
with the adapter) to score the fine-tuned model on the same test set and metrics,
producing the **before/after** benchmark vs the base-model baseline.
