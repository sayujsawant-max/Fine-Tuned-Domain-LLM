# FinSage-7B — Deployment Checklist

A practical pre-flight list for taking the stack from local demo to a public
GPU deployment. See [docs/deployment_guide.md](../docs/deployment_guide.md) for
detail.

## 1. Before deployment

- [ ] Decide the mode: **demo** (no GPU), **full local**, or **GPU VM**.
- [ ] Repo cloned; running from the repo root.
- [ ] Docker + Docker Compose v2 installed (`docker compose version`).
- [ ] For GPU: NVIDIA driver + NVIDIA Container Toolkit installed and verified.

## 2. Environment variables

- [ ] `cp .env.example .env`.
- [ ] `API_SECRET_KEY` set to a strong value (NOT `change-me`).
- [ ] `ENVIRONMENT=production` for any non-local deployment.
- [ ] `CORS_ALLOWED_ORIGINS` restricted to the real frontend origin.
- [ ] `NEXT_PUBLIC_DEMO_MODE=false` for full mode (`true` for demo-only).
- [ ] No secret placed in any `NEXT_PUBLIC_*` variable.

## 3. Model files (full / GPU modes)

- [ ] Merged model present at `checkpoints/finsage-7b-merged` (`make merge-adapter`).
- [ ] Sufficient GPU memory for the chosen `MAX_MODEL_LEN` / `GPU_MEMORY_UTILIZATION`.

## 4. Docker build

- [ ] `make docker-build-all` (or `docker compose ... build`) succeeds.
- [ ] API image does **not** install vLLM/GPU deps (CPU-only).
- [ ] Frontend image builds and does not bake in the API secret.

## 5. Health checks

- [ ] `vllm` container healthy (`GET :8000/v1/models`).
- [ ] `api` container healthy (`GET :8080/v1/health`).
- [ ] `frontend` container healthy (`GET :3000/api/health`).
- [ ] `make check-full-stack` passes (report at `reports/figures/full_stack_health.json`).

## 6. API test

- [ ] `GET /v1/health` returns `ok` (no auth).
- [ ] `GET /v1/ready` reports `vllm_available: true` (full mode).
- [ ] `POST /v1/chat` with a valid `X-API-Key` returns a grounded answer + disclaimer.
- [ ] Invalid/missing key returns `401`; over-limit returns `429`.

## 7. Frontend test

- [ ] <http://localhost:3000> loads; a sample analysis returns an answer.
- [ ] Disclaimer is visible; request ID + latency shown.
- [ ] Browser DevTools shows **no** API secret in any request/bundle.

## 8. Latency benchmark

- [ ] `make benchmark-api` writes `reports/figures/api_latency_benchmark.json`.
- [ ] p50/p95/p99 latencies are acceptable for the demo.

## 9. Security review

- [ ] vLLM (:8000) not exposed publicly (port mapping removed in prod).
- [ ] HTTPS terminated at a reverse proxy / platform.
- [ ] Firewall: only the frontend (and optionally API) port is public.
- [ ] `LOG_REQUEST_BODY=false`; raw filings are not logged.
- [ ] Secrets rotated; `.env` not committed; bundle contains no secrets/weights.
- [ ] External rate limiter considered for multi-replica API.

## 10. Final demo checklist

- [ ] One-line pitch ready (see [reports/demo_script.md](demo_script.md)).
- [ ] Demo mode link works with no backend.
- [ ] Full stack reachable end to end; comparison + benchmark panels render.
- [ ] Graceful failure: stopping vLLM yields a clean `503` / demo fallback, not a crash.
