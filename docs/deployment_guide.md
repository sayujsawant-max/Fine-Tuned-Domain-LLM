# Deployment Guide (Phases 8–11)

## Topology

```
client → FastAPI (public, :8080) → vLLM (internal, :8000) → merged model
```

FastAPI owns auth, request logging, prompt templates, safety checks, and
disclaimer injection. vLLM is never exposed publicly.

## 1. Serve the model with vLLM (Phase 8, GPU)

```bash
make install-serving
make serve-vllm        # serving/vllm_server.sh
```

Serves an OpenAI-compatible API from `MERGED_MODEL_PATH`.

## 2. Run the FastAPI service (Phase 9)

```bash
make serve-api         # uvicorn finsage.serving.app:app
curl http://localhost:8080/v1/health
```

In Phase 1 `/v1/chat` returns a mock response; Phase 9 wires it to the vLLM
backend and adds API-key auth (`API_SECRET_KEY`).

## 3. Docker (Phase 11)

```bash
make docker-build      # docker compose -f docker/docker-compose.yml build
make docker-up
```

- [docker/Dockerfile.api](../docker/Dockerfile.api) — FastAPI service (CPU).
- [docker/Dockerfile.serving](../docker/Dockerfile.serving) — vLLM (GPU, placeholder).
- [docker/Dockerfile.frontend](../docker/Dockerfile.frontend) — demo UI (placeholder).
- `vllm` and `frontend` services are commented out in compose until their phases.

> Behind a corporate TLS-intercepting proxy, bake host root CAs into build images
> or `pip`/`npm` will fail SSL verification.

## 4. Environment

Copy `.env.example` to `.env` and fill in tokens, paths, and `API_SECRET_KEY`.
Never commit `.env`.
