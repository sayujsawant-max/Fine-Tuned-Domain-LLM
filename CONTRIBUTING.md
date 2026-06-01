# Contributing to FinSage-7B

Thanks for your interest! This is a portfolio project, but contributions and
suggestions are welcome.

## Setup

```bash
git clone https://github.com/sayujsawant-max/Fine-Tuned-Domain-LLM.git finsage-7b
cd finsage-7b
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Heavy/GPU extras are optional: `.[ml,training,serving]` (GPU), `.[reporting]`,
`.[llm]`, `.[redis]`, `.[docs]`.

## Test, lint, typecheck

```bash
make check          # ruff + black --check + mypy + pytest
# or directly:
ruff check .
black --check .
mypy
pytest
```

Frontend (optional, needs Node 20+):

```bash
cd frontend && npm install
npm run lint && npm run typecheck && npm run test && npm run build
```

## Branching

- Branch off `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<topic>`.
- Keep PRs focused and small where possible.

## PR checklist

- [ ] `make check` passes (ruff, black, mypy, pytest).
- [ ] Frontend checks pass if you touched `frontend/`.
- [ ] Docs updated for user-facing changes.
- [ ] **No secrets**, `.env`, model weights, or raw data committed.
- [ ] **No mock/sample metrics presented as real results.**
- [ ] `python scripts/final_repo_check.py` passes.

## Code style

- **Python:** ruff + black (line length 100), full type hints, Google-style
  docstrings on public functions/classes. Use `logging` / `rich`, not `print`.
- **TypeScript:** strict mode; `npm run lint` + `typecheck` clean.
- Keep heavy GPU/ML dependencies out of the default install (use optional extras).

## Documentation style

- Markdown, concise and scannable. Recruiter-facing docs short; technical docs
  detailed. Be honest — never overclaim or present unverified numbers as real.

## Security note

Never commit secrets or real API keys. Report vulnerabilities privately — see
[SECURITY.md](SECURITY.md). Do not expose the vLLM server publicly; only the
authenticated FastAPI wrapper should be public.
