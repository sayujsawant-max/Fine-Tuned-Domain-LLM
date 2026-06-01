# FinSage-7B Frontend (Phase 9)

A polished, recruiter-friendly demo for **FinSage-7B** — a Mistral-7B model
fine-tuned on SEC filings. Paste a filing excerpt, pick one of 10 tasks, ask a
question, and get a grounded answer with model, latency, request ID, and the
mandatory disclaimer. Optional base-Mistral comparison and a benchmark summary.

## Stack

Next.js 14 (App Router) · TypeScript (strict) · Tailwind CSS · lucide-react ·
Vitest + Testing Library.

## Security model

The browser **never** holds the API key. It calls a same-origin server route
(`app/api/chat/route.ts`) which injects `X-API-Key` from the server-only
`API_SECRET_KEY` and forwards to the FastAPI backend.

```
Browser → /api/chat (server) → FastAPI (:8080) → vLLM (:8000) → FinSage-7B
```

Only `NEXT_PUBLIC_*` values reach the client; never put secrets there.

## Quickstart

```bash
cp .env.example .env.local
npm install
npm run dev            # http://localhost:3000
```

- **Demo mode (no backend):** `NEXT_PUBLIC_DEMO_MODE=true npm run dev`
- **Live backend:** start vLLM + the API (Phase 8), then
  `API_BASE_URL=http://localhost:8080/v1 API_SECRET_KEY=change-me NEXT_PUBLIC_DEMO_MODE=false npm run dev`

## Scripts

| Command | Purpose |
|---------|---------|
| `npm run dev` | Dev server |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run lint` | ESLint (next lint) |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run test` | Vitest unit/component tests (no backend, no network) |

## Layout

```
app/            layout, page, globals.css, api/chat + api/health proxy routes
components/     FilingInput, TaskSelector, QuestionInput, AnalyzeButton,
                ResponsePanel, ResponseComparison, BenchmarkSummary,
                ArchitectureFlow, StatusBadge, ErrorAlert, CopyButton
lib/            config, taskTypes, samples, api, mockResponses, utils
tests/          taskTypes, api, mockResponses, components
```

Sample filing text is **fabricated** for the demo — not copied from any real SEC
filing. See [reports/demo_script.md](../reports/demo_script.md) for the demo flow.
