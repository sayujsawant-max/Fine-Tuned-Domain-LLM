.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install install-dev install-ml install-training install-serving \
        lint format typecheck test check download-data extract-sections \
        build-dataset validate-dataset train-dry-run train merge-adapter \
        eval-baseline eval-baseline-real eval-finetuned eval-finetuned-adapter \
        eval-finetuned-merged compare-models serve-api serve-vllm \
        docker-build docker-up report

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install: ## Install the lightweight default package
	$(PYTHON) -m pip install -e .

install-dev: ## Install dev tooling (pytest, ruff, black, mypy)
	$(PYTHON) -m pip install -e ".[dev]"

install-ml: ## Install CPU-friendly ML/eval dependencies
	$(PYTHON) -m pip install -e ".[ml]"

install-training: ## Install GPU training stack (torch, peft, trl, bitsandbytes)
	$(PYTHON) -m pip install -e ".[training]"

install-serving: ## Install the vLLM serving stack
	$(PYTHON) -m pip install -e ".[serving]"

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
lint: ## Run ruff linter
	$(PYTHON) -m ruff check .

format: ## Format code with black and apply ruff fixes
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check --fix .

typecheck: ## Run mypy static type checks
	$(PYTHON) -m mypy

test: ## Run the test suite
	$(PYTHON) -m pytest

check: lint typecheck test ## Run lint + typecheck + tests (+ dataset validation if built)
	@if [ -f data/datasets/train.jsonl ]; then \
		$(PYTHON) scripts/validate_dataset.py validate \
			--train-path data/datasets/train.jsonl \
			--validation-path data/datasets/validation.jsonl \
			--test-path data/datasets/test.jsonl \
			--report-path data/datasets/validation_report.json; \
	else echo "No dataset built; skipping dataset validation."; fi

# ---------------------------------------------------------------------------
# Data pipeline (Phase 2-4)
# ---------------------------------------------------------------------------
download-data: ## Download a small set of SEC EDGAR filings
	$(PYTHON) scripts/download_edgar.py download --tickers AAPL MSFT --forms 10-K --start-year 2022 --end-year 2023 --limit-per-company 1

extract-sections: ## Extract sections from downloaded filings into clean text
	$(PYTHON) scripts/extract_sections.py extract --manifest-path data/raw/sec/manifest.jsonl --output-dir data/processed/sec --processed-manifest-path data/processed/sec/manifest.jsonl

build-dataset: ## Build the JSONL instruction dataset
	$(PYTHON) scripts/build_instruction_dataset.py build --processed-manifest-path data/processed/sec/manifest.jsonl --output-dir data/datasets --split-strategy company_holdout

validate-dataset: ## Validate the instruction dataset splits
	$(PYTHON) scripts/validate_dataset.py validate --train-path data/datasets/train.jsonl --validation-path data/datasets/validation.jsonl --test-path data/datasets/test.jsonl --report-path data/datasets/validation_report.json

# ---------------------------------------------------------------------------
# Training & evaluation (Phase 5-7)
# ---------------------------------------------------------------------------
train-dry-run: ## Validate training inputs without loading a model (CPU)
	$(PYTHON) training/train.py --train-file tests/fixtures/train_sample.jsonl --validation-file tests/fixtures/validation_sample.jsonl --output-dir /tmp/finsage_train_dry_run --config configs/training_config.yaml --lora-config configs/lora_config.yaml --dry-run

train: ## Run QLoRA fine-tuning (requires ml,training extras + GPU)
	$(PYTHON) training/train.py --train-file data/datasets/train.jsonl --validation-file data/datasets/validation.jsonl --model-id mistralai/Mistral-7B-Instruct-v0.3 --output-dir checkpoints/finsage-7b --config configs/training_config.yaml --lora-config configs/lora_config.yaml --use-4bit

merge-adapter: ## Merge the trained LoRA adapter into the base model (GPU)
	$(PYTHON) training/merge_adapter.py --base-model mistralai/Mistral-7B-Instruct-v0.3 --adapter-path checkpoints/finsage-7b --output-dir checkpoints/finsage-7b-merged

eval-baseline: ## Baseline eval with the mock backend (no model download)
	$(PYTHON) evaluation/run_baseline_eval.py --test-file data/datasets/test.jsonl --output-dir reports/figures --backend mock --max-examples 50

eval-baseline-real: ## Baseline eval with the real base model (requires ml,training extras + GPU)
	$(PYTHON) evaluation/run_baseline_eval.py --test-file data/datasets/test.jsonl --output-dir reports/figures --backend transformers --model-id mistralai/Mistral-7B-Instruct-v0.3 --max-examples 200 --device auto --load-in-4bit

eval-finetuned: ## Fine-tuned eval with the mock backend (no model download)
	$(PYTHON) evaluation/run_finetuned_eval.py --test-file data/datasets/test.jsonl --baseline-results reports/figures/baseline_results.json --baseline-predictions reports/figures/baseline_predictions.jsonl --output-dir reports/figures --backend mock --max-examples 50

eval-finetuned-adapter: ## Fine-tuned eval with base + LoRA adapter (requires ml,training + GPU)
	$(PYTHON) evaluation/run_finetuned_eval.py --test-file data/datasets/test.jsonl --baseline-results reports/figures/baseline_results.json --baseline-predictions reports/figures/baseline_predictions.jsonl --model-id mistralai/Mistral-7B-Instruct-v0.3 --adapter-path checkpoints/finsage-7b --output-dir reports/figures --backend adapter --device auto --load-in-4bit --max-examples 200

eval-finetuned-merged: ## Fine-tuned eval with the merged model (requires ml,training + GPU)
	$(PYTHON) evaluation/run_finetuned_eval.py --test-file data/datasets/test.jsonl --baseline-results reports/figures/baseline_results.json --baseline-predictions reports/figures/baseline_predictions.jsonl --merged-model-path checkpoints/finsage-7b-merged --output-dir reports/figures --backend merged --device auto --max-examples 200

compare-models: ## Compare existing baseline/fine-tuned outputs into a benchmark report
	$(PYTHON) evaluation/compare_models.py --baseline-results reports/figures/baseline_results.json --baseline-predictions reports/figures/baseline_predictions.jsonl --finetuned-results reports/figures/finetuned_results.json --finetuned-predictions reports/figures/finetuned_predictions.jsonl --output-dir reports/figures --report-path reports/benchmark_report.md

report: compare-models ## Alias for compare-models

# ---------------------------------------------------------------------------
# Serving (Phase 8-9)
# ---------------------------------------------------------------------------
serve-api: ## Run the FastAPI service
	$(PYTHON) -m uvicorn finsage.serving.app:app --host $${API_HOST:-localhost} --port $${API_PORT:-8080}

serve-vllm: ## Start the vLLM OpenAI-compatible server
	bash serving/vllm_server.sh

# ---------------------------------------------------------------------------
# Docker (Phase 11)
# ---------------------------------------------------------------------------
docker-build: ## Build Docker images
	docker compose -f docker/docker-compose.yml build

docker-up: ## Start the Docker stack
	docker compose -f docker/docker-compose.yml up
