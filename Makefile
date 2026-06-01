.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install install-dev install-ml install-training install-serving \
        lint format typecheck test check download-data extract-sections \
        build-dataset validate-dataset train eval-baseline eval-finetuned serve-api \
        serve-vllm docker-build docker-up report

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

check: lint typecheck test ## Run lint + typecheck + tests

# ---------------------------------------------------------------------------
# Data pipeline (Phase 2-4)
# ---------------------------------------------------------------------------
download-data: ## Download a small set of SEC EDGAR filings
	$(PYTHON) scripts/download_edgar.py download --tickers AAPL MSFT --forms 10-K --start-year 2022 --end-year 2023 --limit-per-company 1

extract-sections: ## Extract sections from downloaded filings into clean text
	$(PYTHON) scripts/extract_sections.py extract --manifest-path data/raw/sec/manifest.jsonl --output-dir data/processed/sec --processed-manifest-path data/processed/sec/manifest.jsonl

build-dataset: ## Build the JSONL instruction dataset (placeholder)
	$(PYTHON) scripts/build_instruction_dataset.py run

validate-dataset: ## Validate the instruction dataset (placeholder)
	$(PYTHON) scripts/validate_dataset.py run

# ---------------------------------------------------------------------------
# Training & evaluation (Phase 5-7)
# ---------------------------------------------------------------------------
train: ## Run QLoRA fine-tuning (requires training extras)
	$(PYTHON) training/train.py

eval-baseline: ## Evaluate the base model (requires ml extras)
	$(PYTHON) evaluation/run_baseline_eval.py

eval-finetuned: ## Evaluate the fine-tuned model (requires ml extras)
	$(PYTHON) evaluation/run_finetuned_eval.py

report: ## Generate the benchmark comparison report
	$(PYTHON) evaluation/compare_models.py

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
