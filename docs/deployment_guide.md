# Deployment Guide

## Topology

```
Browser → Frontend (:3000) → Next.js /api proxy → FastAPI (:8080) → vLLM (:8000, internal) → merged model
```

Phase 7 delivers the **vLLM inference engine**; Phase 8 the public **FastAPI
wrapper** (auth, logging, prompt templates, disclaimer, rate limiting); Phase 9
the **Next.js frontend**; Phase 10 ties it together with **Docker Compose**. The
vLLM server is **internal-only** and must never be exposed publicly — all public
traffic flows through the frontend and API.

## Phase 10 — full-stack deployment

### 1. Deployment architecture

Three Docker Compose files compose into the modes below. Services share a private
`finsage-network`; `api` waits for vLLM health, `frontend` waits for API health.

| File | Adds |
|------|------|
| `docker/docker-compose.yml` | base stack: `vllm`, `api`, `frontend` |
| `docker/docker-compose.demo.yml` | GPU-free `api-demo` + `frontend-demo` (demo mode) |
| `docker/docker-compose.gpu.yml` | NVIDIA device reservations for `vllm` |

### 2. Local demo deployment (no GPU, no model)

```bash
cp .env.example .env
make deploy-demo
# or: docker compose -f docker/docker-compose.demo.yml up --build
```

Open <http://localhost:3000>. The frontend runs with `NEXT_PUBLIC_DEMO_MODE=true`;
when the backend can't serve a real answer the `/api/chat` proxy returns a
clearly-labelled mock. Ideal for recruiters and laptops.

### 3. Local full-stack deployment (GPU + merged model)

```bash
cp .env.example .env                 # set a strong API_SECRET_KEY
make merge-adapter                   # produce checkpoints/finsage-7b-merged (once)
make deploy-full                     # docker compose -f docker/docker-compose.yml up --build
```

`vllm` mounts the merged model read-only at `/models/finsage-7b-merged`. First
start is slow (7B weights); watch `GET :8000/v1/models`.

### 4. GPU VM deployment (RunPod / Lambda / any provider)

```bash
# On the GPU VM, from the repo root:
bash scripts/deploy_gpu_vm.sh
# or explicitly:
make deploy-gpu                      # base + gpu override
```

`deploy_gpu_vm.sh` checks Docker + the NVIDIA Container Toolkit, prepares `.env`,
and starts the GPU stack. Verify GPU visibility first:
`docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`.
Package a provider-agnostic deployment bundle with `make export-deployment`
(→ `dist/finsage-deployment-bundle/`; no weights, data, or secrets).

### 5. Docker Compose services

| Service | Image | Port | Notes |
|---------|-------|------|-------|
| `vllm` | `Dockerfile.serving` (CUDA) | 8000 | Internal; GPU; mounts merged model RO |
| `api` | `Dockerfile.api` (slim, non-root) | 8080 | No vLLM deps; `depends_on` vllm healthy |
| `frontend` | `Dockerfile.frontend` (node:20-alpine) | 3000 | `depends_on` api healthy; secret server-side |

### 6. Environment variables

See `.env.example`. Server-only (never `NEXT_PUBLIC_`): `API_SECRET_KEY`,
`API_BASE_URL`. Compose interpolates `${API_SECRET_KEY:-change-me}` from your
shell or `.env`. vLLM tunables: `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`,
`TENSOR_PARALLEL_SIZE`, `MAX_NUM_SEQS`, `DTYPE`.

### 7. Health checks

```bash
make check-full-stack    # frontend + API (/health,/ready,/chat) + vLLM
# demo (skip vLLM):
python scripts/check_full_stack.py --demo
```

Writes `reports/figures/full_stack_health.json`; exits non-zero on failure. Each
container also has a Docker `HEALTHCHECK`, and Compose gates startup on them.

### 8. Latency benchmarking

```bash
make benchmark-api       # through the FastAPI wrapper (api_chat)
make benchmark-vllm      # directly against vLLM (vllm_chat_completions)
```

Reports p50/p95/p99, average/min/max, and (vLLM) approximate tokens/sec to
`reports/figures/*_latency_benchmark.json`.

### 9. Security checklist

- [ ] Set a strong `API_SECRET_KEY` (never `change-me`). For rotation, issue
      multiple keys at once: `API_SECRET_KEY=new-key,old-key` (comma-separated).
- [ ] vLLM is **internal**: the base compose binds :8000 to **loopback only**
      (`127.0.0.1`), so it is unreachable off-host. For full isolation, delete the
      `ports` mapping entirely (the api reaches it via `http://vllm:8000`).
- [ ] Serve over **HTTPS** (reverse proxy / platform TLS).
- [ ] Restrict `CORS_ALLOWED_ORIGINS` to the real frontend origin(s).
- [ ] Keep `LOG_REQUEST_BODY=false` — never log raw filings.
- [ ] Put the stack behind a reverse proxy + firewall; expose only :3000 (and :8080 if needed).
- [ ] For multi-replica API, set `RATE_LIMIT_BACKEND=redis` + `REDIS_URL` and
      install the `redis` extra (`pip install -e ".[redis]"`). The limiter falls
      back to in-memory (per-process) if Redis is unavailable.
- [ ] Monitor GPU memory; tune `GPU_MEMORY_UTILIZATION` / `MAX_MODEL_LEN`.
- [ ] Match the vLLM CUDA base to your driver/wheel:
      `--build-arg CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
- [ ] Add authentication in front of the frontend if it is public.
- [ ] Confirm the browser bundle holds no secret (`grep -r NEXT_PUBLIC frontend`).
- [ ] Populate the benchmark panel with real metrics via the frontend
      `BENCHMARK_DATA` env var once a real evaluation has run.

### 10. Public demo deployment options

- **Demo mode anywhere:** deploy only the frontend (Vercel / any Node host) with
  `NEXT_PUBLIC_DEMO_MODE=true` — no GPU, mocked answers, recruiter-ready.
- **Frontend on Vercel + API/vLLM on a GPU VM:** set `API_BASE_URL` (server-only)
  to the HTTPS API endpoint and `API_SECRET_KEY` as an encrypted env var.
- **All-in-one GPU VM:** `make deploy-gpu` behind a TLS reverse proxy.

### 11. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Merged model not found` | Run `make merge-adapter` or set `MODEL_PATH`. |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing/not configured; `nvidia-ctk runtime configure`. |
| vLLM slow to become healthy | 7B weights load for minutes; the healthcheck `start_period` is 300s. |
| API cannot reach vLLM | Use `VLLM_BASE_URL=http://vllm:8000/v1` inside Compose; check the vllm logs. |
| Frontend cannot reach API | Use `API_BASE_URL=http://api:8080/v1` inside Compose. |
| `CORS error` in the browser | Add the frontend origin to `CORS_ALLOWED_ORIGINS`. |
| `401 Unauthorized` | Send a valid `X-API-Key`; in production set a real `API_SECRET_KEY`. |
| `port is already allocated` | Stop the conflicting process or change the published port. |
| Docker GPU runtime error | Verify with `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`. |

## Phase 8 — FastAPI wrapper

The wrapper (`finsage.serving.app:create_app`) is a thin, CPU-only service that
proxies to vLLM over HTTP and layers on production concerns:

| Concern | Where | Notes |
|---------|-------|-------|

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

## Phase 9 — frontend deployment

The [Next.js frontend](../frontend/) is a CPU-only static UI plus two server-side
proxy routes (`/api/chat`, `/api/health`). **The API secret never reaches the
browser** — it is read server-side from `API_SECRET_KEY` and injected as the
`X-API-Key` header by the proxy. Anything prefixed `NEXT_PUBLIC_*` is public.

### Environment variables

| Variable | Scope | Purpose |
|----------|-------|---------|
| `API_BASE_URL` | server-only | Backend base URL the proxy forwards to (e.g. `http://api:8080/v1`) |
| `API_SECRET_KEY` | server-only | Injected as `X-API-Key`; never bundled into client code |
| `NEXT_PUBLIC_API_BASE_URL` | public | Display / health only |
| `NEXT_PUBLIC_DEMO_MODE` | public | `true` → return a labelled mock when the backend is unreachable |
| `NEXT_PUBLIC_APP_NAME` | public | App title |

### Demo mode vs real backend mode

- **Demo mode** (`NEXT_PUBLIC_DEMO_MODE=true`): if the backend is down, the proxy
  returns a deterministic mock clearly marked "Demo mode response." Ideal for
  static hosting and recruiter links with no GPU/backend running.
- **Real backend mode** (`NEXT_PUBLIC_DEMO_MODE=false`): the proxy forwards to the
  FastAPI wrapper and surfaces real errors (401/429/503) instead of mocking.

### Vercel (or any Node host)

1. Set the project root to `frontend/`.
2. Configure env vars: `API_BASE_URL`, `API_SECRET_KEY` (as **encrypted/secret**,
   not `NEXT_PUBLIC_*`), and `NEXT_PUBLIC_DEMO_MODE`.
3. Deploy. The `/api/*` routes run as serverless functions and hold the secret
   server-side. Point `API_BASE_URL` at your **HTTPS** FastAPI endpoint.

### Docker

```bash
make docker-build-frontend         # docker build -f docker/Dockerfile.frontend -t finsage-frontend:latest frontend
make docker-up-frontend            # starts api + vllm via depends_on
```

The compose `frontend` service publishes `:3000`, sets `API_BASE_URL=http://api:8080/v1`
and `API_SECRET_KEY` (server-side), and `depends_on` the API being healthy. Only
the frontend port should be exposed publicly; keep the API and vLLM on the
private network.

### Frontend production notes

- Serve over **HTTPS** (Vercel/edge or a reverse proxy with TLS).
- Set a **real `API_SECRET_KEY`** server-side; never as `NEXT_PUBLIC_*`.
- **Restrict CORS** on the FastAPI side to the frontend origin.
- **Do not expose vLLM** — only the frontend (and optionally the API) is public.
- The browser bundle contains no secret; verify with `grep -r NEXT_PUBLIC frontend`
  and confirm no key material is referenced client-side.

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
