# Deployment Guide

## Topology

```
client → FastAPI wrapper (public, :8080, Phase 8) → vLLM (internal, :8000, Phase 7) → merged model
```

Phase 7 delivers the **vLLM inference engine**; Phase 8 delivers the public
**FastAPI wrapper** (auth, logging, prompt templates, disclaimer injection, rate
limiting). The vLLM server is **internal-only** and must never be exposed to the
public internet — all public traffic goes through the FastAPI wrapper.

## Phase 8 — FastAPI wrapper

The wrapper (`finsage.serving.app:create_app`) is a thin, CPU-only service that
proxies to vLLM over HTTP and layers on production concerns:

| Concern | Where | Notes |
|---------|-------|-------|
| **Auth** | `finsage.serving.auth` | `X-API-Key` or `Authorization: Bearer`; constant-time compare. Dev allows the `change-me` placeholder with a warning; production rejects it. Public paths: `/v1/health`, `/docs`, `/openapi.json`, `/redoc`. |
| **Rate limiting** | `finsage.serving.rate_limiter` + middleware | In-memory sliding window per client (API-key hash or IP). HTTP 429 + `X-RateLimit-*` headers. Single-process only. |
| **Logging** | `StructuredLoggingMiddleware` | One JSON line per request: request_id, method, path, status, latency, client host, user agent, input **char count only** — never filing text (unless `LOG_REQUEST_BODY=true`). |
| **Request IDs** | `RequestIDMiddleware` | UUID per request, echoed as `X-Request-ID`, included in every error body. |
| **Security headers** | `SecurityHeadersMiddleware` | `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`. |
| **Disclaimer** | `finsage.serving.disclaimer` | Appended (deduplicated) when `DISCLAIMER_ENABLED=true`. |
| **Errors** | `finsage.serving.errors` | Uniform `{error, detail, request_id}`: 422 validation, 401 auth, 429 rate limit, 503 vLLM down, 500 unexpected. |

### Run

```bash
# vLLM (GPU host):
MODEL_PATH=checkpoints/finsage-7b-merged SERVED_MODEL_NAME=finsage-7b bash serving/vllm_server.sh
# API (CPU):
API_SECRET_KEY=change-me VLLM_BASE_URL=http://localhost:8000/v1 bash serving/start_api.sh
python scripts/check_api_server.py --base-url http://localhost:8080/v1 --api-key change-me
```

### Docker Compose (API + vLLM)

`docker/docker-compose.yml` defines `api` (CPU, `Dockerfile.api`, **no vLLM/GPU
deps**) and `vllm` (GPU). `api` waits for `vllm` to become healthy
(`depends_on: condition: service_healthy`) and points at it via
`VLLM_BASE_URL=http://vllm:8000/v1`. Only the API port (8080) should be published
externally.

```bash
make docker-build-api       # CPU image
make docker-up-full         # vLLM + API
```

### Production considerations

- **Set a real `API_SECRET_KEY`** and `ENVIRONMENT=production` (the placeholder
  is rejected in production).
- **Terminate TLS / use HTTPS** at a reverse proxy (nginx, Traefik, a cloud LB)
  in front of the API.
- **Restrict CORS** via `CORS_ALLOWED_ORIGINS` to known frontends — do not use
  `*` with credentials.
- **Use an external rate limiter** (e.g. Redis) for multi-replica deployments;
  the in-memory limiter is per-process only.
- **Put the API behind a reverse proxy** and keep vLLM on a private network.
- **Do not log raw filings** — keep `LOG_REQUEST_BODY=false` (the default).
- Rotate API keys; never commit `.env` or `api_keys.txt`.

## Why vLLM

vLLM provides high-throughput batched inference and an **OpenAI-compatible** API
(`/v1/models`, `/v1/chat/completions`), so any OpenAI-style client works and the
Phase 8 wrapper stays thin. PagedAttention keeps GPU memory efficient for 7B.

## Merged model vs adapter serving

- **Merged model (recommended):** merge the LoRA adapter into the base weights
  (`training/merge_adapter.py`) and serve the standalone model
  (`serving/vllm_server.sh`). Simplest and most portable.
- **Direct LoRA (optional):** `serving/vllm_lora_server.sh` serves the base model
  with `--enable-lora`/`--lora-modules`. Flags vary across vLLM versions; verify
  with `vllm serve --help`. Prefer the merged path for Phase 7.

## Local GPU serving

```bash
pip install -e ".[serving]"           # CUDA GPU + Linux/WSL2
python training/merge_adapter.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter-path checkpoints/finsage-7b --output-dir checkpoints/finsage-7b-merged
MODEL_PATH=checkpoints/finsage-7b-merged bash serving/vllm_server.sh
```

Tunables (env vars / `configs/serving_config.yaml`): `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`, `MAX_NUM_SEQS`, `DTYPE`.

## Docker serving (GPU)

```bash
docker build -f docker/Dockerfile.serving -t finsage-vllm:latest .
docker compose -f docker/docker-compose.yml up vllm
```

Requires an NVIDIA GPU + the **NVIDIA Container Toolkit**. The merged model is
mounted read-only at `/models/finsage-7b-merged`. If your Docker/Compose does not
support `deploy.resources` GPU reservations, run with `docker run --gpus all` or
set the default Docker runtime to `nvidia`. vLLM is pinned to a CUDA base image;
if `pip install vllm` resolves a wheel for a different CUDA, match the base tag
or use the official `vllm/vllm-openai` image and mount the model.

## OpenAI-compatible API usage

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "finsage-7b",
  "messages": [
    {"role":"system","content":"You are FinSage-7B, a financial filing analysis assistant. Do not provide investment advice."},
    {"role":"user","content":"Summarize the key risk factors: competition, supply chain disruption, regulatory uncertainty."}
  ],
  "temperature": 0.0, "max_tokens": 256 }'
```

Or use the bundled client: `from finsage.serving import VLLMClient`.

## Health checks

- `GET /v1/models` is the readiness probe (used by the Docker `healthcheck`).
- `finsage.serving.health.wait_for_vllm(base_url, timeout, poll_interval)` blocks
  until ready (7B weights can take minutes to load).
- `python serving/test_endpoint.py all ...` runs a health + chat smoke test.

## Latency benchmarking

```bash
python serving/benchmark_latency.py --base-url http://localhost:8000/v1 \
  --model finsage-7b --num-requests 20 --concurrency 1 \
  --output-path reports/figures/vllm_latency_benchmark.json
```

Reports p50/p95/p99, average/min/max latency, and approximate tokens/sec.

## ⚠️ Security note

The vLLM endpoint has **no authentication, rate limiting, or disclaimer
injection** (the optional `--api-key` is a single shared secret only). **Do not
expose it to the public internet.** The Phase 8 FastAPI wrapper owns auth,
request logging, prompt templates, safety checks, and mandatory disclaimer
injection, and is the only surface intended to be public — publish only its port
(8080) and keep vLLM (8000) on a private network.

## Environment

Copy `.env.example` to `.env`. vLLM vars: `MODEL_PATH`, `SERVED_MODEL_NAME`,
`VLLM_HOST`, `VLLM_PORT`, `VLLM_BASE_URL`, `VLLM_API_KEY`, `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`, `MAX_NUM_SEQS`. Never commit `.env`.
Behind a corporate TLS-intercepting proxy, bake host root CAs into build images.
