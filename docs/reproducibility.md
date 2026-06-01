# FinSage-7B — Reproducibility Guide

Everything except real training runs CPU-only and offline (mock backends). No GPU,
no model downloads, and no external API calls are required to validate the
pipeline.

## 1. Environment setup

- Python 3.11+ (3.12 supported). Node 20+ for the frontend.
- Optional: Docker + Docker Compose; an NVIDIA GPU for real training/serving.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extra dependency groups: `ml`, `training`, `serving` (GPU), `llm`,
`reporting`, `redis`, `docs`.

## 2. Frontend dependencies

```bash
cd frontend
npm install        # on this machine, prefix with NODE_OPTIONS=--use-system-ca if behind a TLS proxy
```

## 3. Quality gates

```bash
make check         # ruff + black --check + mypy + pytest
# or directly:
ruff check . && black --check . && mypy && pytest
```

## 4. Dataset generation

```bash
python scripts/download_edgar.py download --tickers AAPL MSFT --forms 10-K
python scripts/extract_sections.py extract
python scripts/build_instruction_dataset.py build
python scripts/validate_dataset.py validate
```

## 5. Baseline evaluation

```bash
# Mock backend (CPU-only) — validates the eval pipeline:
python evaluation/run_baseline_eval.py --backend mock
# Real base model (GPU): --backend transformers --model-id mistralai/Mistral-7B-Instruct-v0.3
```

## 6. Training dry-run (CPU)

```bash
python training/train.py --dry-run \
    --train-file data/datasets/train.jsonl \
    --validation-file data/datasets/validation.jsonl \
    --config configs/training_config.yaml --lora-config configs/lora_config.yaml
```

## 7. Real training (GPU)

```bash
pip install -e ".[ml,training]"
python training/train.py --train-file data/datasets/train.jsonl ...   # saves to checkpoints/finsage-7b
python training/merge_adapter.py                                       # merge adapter into base
```

## 8. Fine-tuned evaluation + comparison

```bash
python evaluation/run_finetuned_eval.py --backend mock        # or adapter/merged (GPU)
python evaluation/compare_models.py                           # writes reports/figures/*
```

## 9. Report generation

```bash
pip install -e ".[reporting]"     # matplotlib, markdown, reportlab
make report            # reports/benchmark_report.{md,pdf,html} + charts
make validate-report
make report-mock       # clearly-labelled sample report from fixtures
```

## 10. Docker deployment

```bash
cp .env.example .env
make deploy-demo       # frontend + mock API (no GPU)
make deploy-local      # full stack
make deploy-gpu        # NVIDIA Container Toolkit required
```

## 11. Expected outputs

- `data/datasets/{train,validation,test}.jsonl`, `dataset_stats.json`,
  `validation_report.json`
- `reports/figures/*.json` (eval artifacts), `reports/benchmark_report.md` + PDF/HTML
- `checkpoints/finsage-7b/` (adapter + `training_summary.json`) — **GPU only**
- All checks green: `make check` and `python scripts/final_repo_check.py`

## 12. Known platform issues

- **Windows:** `make` may be absent — run the Makefile commands directly. Behind a
  corporate TLS proxy, prefix npm with `NODE_OPTIONS=--use-system-ca`.
- **CUDA / bitsandbytes:** GPU + matching CUDA toolchain required for 4-bit
  training; not installable on most macOS/Windows machines.
- **vLLM:** Linux + CUDA only; on Windows use WSL2. 6 GB VRAM is tight for a 7B
  model — prefer a rented 24 GB GPU (RunPod/Lambda) for real runs.
- **Docker behind a TLS-intercepting proxy:** bake host root CAs into build images
  or pip/npm will fail SSL verification.
