# Security Policy

## Supported versions

This is an actively developed portfolio project; only the latest `main` is
supported.

| Version | Supported |
| --- | --- |
| `main` (latest) | ✅ |
| older commits | ❌ |

## Reporting a vulnerability

Please report security issues **privately**, not via public issues:

- Use GitHub’s **“Report a vulnerability”** (Security Advisories) on the repo, or
- Email the maintainer listed on the GitHub profile.

Include reproduction steps and impact. You’ll get an acknowledgement and a fix
timeline. Please allow reasonable time to remediate before any public disclosure.

## Secret handling

- **Never commit secrets.** `.env`, `frontend/.env.local`, tokens, and API keys
  are git-ignored — keep them that way.
- The default `API_SECRET_KEY` is the placeholder `change-me`; the API **rejects**
  it in production (`ENVIRONMENT=production`) and only warns in development.
- Set a strong, unique `API_SECRET_KEY` (and `HF_TOKEN` / `WANDB_API_KEY` if used)
  via environment variables in your deployment platform.
- `python scripts/final_repo_check.py` scans for hardcoded real secrets.

## API key warning

API-key auth (`X-API-Key`) is **demo-grade** security suitable for a portfolio
deployment — not enterprise SSO/OAuth. For production, place the service behind a
proper API gateway / identity provider.

## vLLM exposure warning

The vLLM inference server has **no authentication**. **Never expose it publicly.**
Only the FastAPI wrapper (with auth, rate limiting, and disclaimer injection)
should be public; vLLM must stay on an internal network / loopback.

## Data privacy

FinSage-7B uses **only public SEC EDGAR filings** — no private or personal data.
Do not feed confidential documents into a publicly hosted instance. Request bodies
(filing text) are **not** logged; only character counts are.

## Financial disclaimer

FinSage-7B is **not** a licensed financial advisor. Outputs are not investment
recommendations and may be incomplete or incorrect. Always verify against the
original filing.
