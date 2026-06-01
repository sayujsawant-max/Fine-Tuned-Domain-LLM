# Training Guide (Phase 6)

## Install the training stack (GPU)

```bash
make install-training   # torch, peft, trl, bitsandbytes, accelerate, wandb
```

> Not part of the default install. Best run on a single 24 GB GPU (RTX 3090/4090)
> or an A100 80 GB on RunPod. `bitsandbytes` on native Windows is painful — prefer
> a Linux box, WSL2, or a cloud GPU.

## Configure

- LoRA: [../configs/lora_config.yaml](../configs/lora_config.yaml)
  (r=16, alpha=32, dropout=0.05, attention + MLP projections, 4-bit NF4).
- Training: [../configs/training_config.yaml](../configs/training_config.yaml)
  (3 epochs, lr 2e-4 cosine, bs 2 × grad-accum 8, bf16, grad checkpointing).
- Set `HF_TOKEN` and `WANDB_API_KEY` in `.env`.

## Run

```bash
make train                      # python training/train.py
python training/merge_adapter.py  # merge LoRA into base for serving
```

## Method

QLoRA: the base model is loaded in 4-bit NF4; only the LoRA adapter is trained
via TRL's `SFTTrainer`. The merged checkpoint goes to `MERGED_MODEL_PATH` for vLLM.

## Tips

- Validate the dataset before training (`make validate-dataset`).
- Watch eval loss; early stopping is configured via `finsage.training.callbacks`.
- Keep the adapter (small) under version control consideration; the merged model
  is large and git-ignored.
