.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install install-dev install-ml install-training install-serving \
        install-llm lint format typecheck test check download-data extract-sections \
        build-dataset validate-dataset enhance-dataset train-dry-run train merge-adapter \
        eval-baseline eval-baseline-real eval-finetuned eval-finetuned-adapter \
        eval-finetuned-merged compare-models serve-api serve-vllm serve-vllm-lora \
        test-vllm test-api-server benchmark-vllm docker-build docker-up docker-build-serving \
        docker-up-serving docker-build-api docker-up-api docker-up-full report \
        frontend-install frontend-dev frontend-build frontend-lint frontend-test \
        frontend-typecheck docker-build-frontend docker-up-frontend check-full \
        deploy-demo deploy-full deploy-gpu check-full-stack benchmark-api \
        export-deployment docker-build-all docker-up-demo

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

install-llm: ## Install the Anthropic SDK for LLM-assisted dataset enhancement
	$(PYTHON) -m pip install -e ".[llm]"

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

check-full: check ## Backend check + frontend checks + sample report (requires frontend deps)
	@if [ -d frontend/node_modules ]; then \
		$(MAKE) frontend-lint frontend-typecheck frontend-test; \
	else echo "frontend/node_modules missing; run 'make frontend-install' first."; fi
	$(MAKE) report-mock

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

enhance-dataset: ## LLM-assist stronger targets (mock; add ANTHROPIC_API_KEY + --no-mock for real)
	$(PYTHON) scripts/enhance_dataset.py enhance --input-path data/datasets/train.jsonl --output-path data/datasets/train_enhanced.jsonl --mock

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

report: ## Generate the polished benchmark report from real artifacts (Phase 11)
	$(PYTHON) scripts/generate_benchmark_report.py generate \
		--input-dir reports/figures \
		--output-dir reports \
		--generate-charts \
		--export-pdf \
		--export-html

report-mock: ## Generate a clearly-labelled sample report from fixtures (Phase 11)
	$(PYTHON) scripts/generate_benchmark_report.py generate \
		--input-dir tests/fixtures/reporting \
		--output-dir /tmp/finsage_report_test \
		--dataset-stats-path tests/fixtures/reporting/dataset_stats_sample.json \
		--training-summary-path tests/fixtures/reporting/training_summary_sample.json \
		--mock-mode \
		--generate-charts \
		--export-html \
		--no-export-pdf

validate-report: ## Validate the generated benchmark report (Phase 11)
	$(PYTHON) scripts/validate_report.py \
		--report-path reports/benchmark_report.md \
		--metadata-path reports/report_metadata.json

# ---------------------------------------------------------------------------
# Serving (Phase 7: vLLM)
# ---------------------------------------------------------------------------
serve-api: ## Run the FastAPI wrapper (Phase 8)
	bash serving/start_api.sh

test-api-server: ## Smoke-test a running API server
	$(PYTHON) scripts/check_api_server.py --base-url http://localhost:8080/v1 --api-key change-me

serve-vllm: ## Start the vLLM OpenAI-compatible server (merged model, GPU)
	bash serving/vllm_server.sh

serve-vllm-lora: ## Serve base model + LoRA adapter directly (optional, GPU)
	bash serving/vllm_lora_server.sh

test-vllm: ## Smoke-test a running vLLM endpoint
	$(PYTHON) serving/test_endpoint.py all --base-url http://localhost:8000/v1 --model finsage-7b

benchmark-vllm: ## Benchmark vLLM latency -> reports/figures/vllm_latency_benchmark.json
	$(PYTHON) serving/benchmark_latency.py --base-url http://localhost:8000/v1 --model finsage-7b --num-requests 20 --concurrency 1 --output-path reports/figures/vllm_latency_benchmark.json

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build Docker images
	docker compose -f docker/docker-compose.yml build

docker-up: ## Start the Docker stack
	docker compose -f docker/docker-compose.yml up

docker-build-serving: ## Build the vLLM serving image (GPU)
	docker build -f docker/Dockerfile.serving -t finsage-vllm:latest .

docker-up-serving: ## Start only the vLLM service (GPU)
	docker compose -f docker/docker-compose.yml up vllm

docker-build-api: ## Build the FastAPI wrapper image (CPU)
	docker build -f docker/Dockerfile.api -t finsage-api:latest .

docker-up-api: ## Start the API service (starts vLLM via depends_on)
	docker compose -f docker/docker-compose.yml up api

docker-up-full: ## Start the full stack (vLLM + API + frontend)
	docker compose -f docker/docker-compose.yml up

# ---------------------------------------------------------------------------
# Frontend (Phase 9: Next.js demo)
# ---------------------------------------------------------------------------
frontend-install: ## Install frontend npm dependencies
	cd frontend && npm install

frontend-dev: ## Run the Next.js dev server (http://localhost:3000)
	cd frontend && npm run dev

frontend-build: ## Build the production frontend
	cd frontend && npm run build

frontend-lint: ## Lint the frontend
	cd frontend && npm run lint

frontend-test: ## Run frontend unit tests (vitest)
	cd frontend && npm run test

frontend-typecheck: ## Type-check the frontend (tsc --noEmit)
	cd frontend && npm run typecheck

docker-build-frontend: ## Build the frontend Docker image
	docker build -f docker/Dockerfile.frontend -t finsage-frontend:latest frontend

docker-up-frontend: ## Start the frontend service (starts api+vllm via depends_on)
	docker compose -f docker/docker-compose.yml up frontend

# ---------------------------------------------------------------------------
# Deployment (Phase 10: full-stack Docker)
# ---------------------------------------------------------------------------
deploy-demo: ## Start the CPU-only demo stack (no GPU/model)
	bash scripts/deploy_local.sh --demo

deploy-full: ## Start the full stack (vllm + api + frontend; requires merged model)
	bash scripts/deploy_local.sh --full

deploy-gpu: ## Start the full stack with explicit GPU reservations
	bash scripts/deploy_local.sh --gpu

check-full-stack: ## Verify frontend + API + vLLM health
	$(PYTHON) scripts/check_full_stack.py \
		--frontend-url http://localhost:3000 \
		--api-url http://localhost:8080/v1 \
		--vllm-url http://localhost:8000/v1 \
		--api-key $${API_SECRET_KEY:-change-me}

benchmark-api: ## Benchmark API /chat latency -> reports/figures/api_latency_benchmark.json
	$(PYTHON) serving/benchmark_latency.py \
		--base-url http://localhost:8080/v1 \
		--endpoint api_chat \
		--api-key $${API_SECRET_KEY:-change-me} \
		--num-requests 20 --concurrency 1 \
		--output-path reports/figures/api_latency_benchmark.json

export-deployment: ## Package deployment files into dist/finsage-deployment-bundle/
	bash scripts/export_deployment_bundle.sh

docker-build-all: ## Build all stack images
	docker compose -f docker/docker-compose.yml build

docker-up-demo: ## Start the demo compose stack directly
	docker compose -f docker/docker-compose.demo.yml up
