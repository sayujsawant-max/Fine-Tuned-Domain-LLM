# FinSage-7B — Final Release Checklist

Run `python scripts/final_repo_check.py` and `make release-check`, then walk this
list before publishing.

## Code quality
- [ ] `ruff check .` clean
- [ ] `black --check .` clean
- [ ] `mypy` clean
- [ ] No dead code / debug prints

## Tests
- [ ] `pytest` green (backend)
- [ ] `cd frontend && npm run test` green (if Node installed)
- [ ] CI green on `main`

## Docs
- [ ] README portfolio-ready; badges resolve
- [ ] project_summary, interview_guide, reproducibility, publishing_guide present
- [ ] model_card + dataset_card present
- [ ] roadmap + faq present

## Frontend
- [ ] `npm run lint` / `typecheck` / `build` pass
- [ ] Demo mode works with no backend
- [ ] No secret in client bundle (`API_SECRET_KEY` server-only)

## Backend
- [ ] `/v1/health` and `/v1/ready` work
- [ ] Auth rejects bad keys in production mode
- [ ] Rate limiting + disclaimer injection verified

## Docker
- [ ] `docker compose config` valid for demo / full / gpu
- [ ] vLLM bound internally (not published)

## Benchmark report
- [ ] `make report` + `make validate-report` pass
- [ ] PDF opens; charts render

## Security
- [ ] `.env` not committed; real `API_SECRET_KEY` set in deploy env
- [ ] vLLM not exposed publicly
- [ ] No raw data / model weights committed

## Honesty check (mandatory)
- [ ] **No mock numbers presented as real**
- [ ] **README states real training status** (pending GPU)
- [ ] **Disclaimer present**
- [ ] **API secret not exposed**
- [ ] **vLLM not exposed publicly**
- [ ] **Raw data / model weights not committed**
- [ ] **Dataset / model cards ready**
- [ ] **Demo mode clearly labeled**

## GitHub polish
- [ ] Description + topics set; repo pinned
- [ ] Issue/PR templates, CONTRIBUTING, SECURITY, LICENSE present
- [ ] `python scripts/final_repo_check.py` passes

## Hugging Face publishing
- [ ] (After real training) adapter/model uploaded
- [ ] Model card numbers are real; template/pending banner removed
- [ ] Dataset card source terms confirmed

## LinkedIn launch
- [ ] Use post Version A until real training; B only with real numbers
- [ ] Demo clip attached; GitHub linked
