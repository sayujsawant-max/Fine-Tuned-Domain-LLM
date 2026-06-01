# FinSage-7B Frontend (Phase 10)

> **Status: placeholder.** The demo UI is built in Phase 10.

## Planned

A lightweight demo where a user pastes a 10-K excerpt, asks a question, and sees:

- the **base Mistral-7B** answer,
- the **FinSage-7B** fine-tuned answer,
- a grounded, structured response with the mandatory disclaimer,
- a benchmark table showing the improvement.

## Likely stack

Streamlit (fastest path) or a Next.js app calling the FastAPI backend at
`/v1/chat`. The choice is finalised in Phase 10.

## Running (future)

The frontend will be added to [docker/docker-compose.yml](../docker/docker-compose.yml)
as the `frontend` service.
