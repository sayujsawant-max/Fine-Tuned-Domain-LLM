# FinSage-7B — Publishing Guide

How to take the repo public across GitHub, Hugging Face, a demo, LinkedIn, and a
resume — **honestly**.

## GitHub checklist

- [ ] `python scripts/final_repo_check.py` passes (no critical failures).
- [ ] `make check` green; CI green on `main`.
- [ ] README renders well; badges resolve; disclaimer + limitations present.
- [ ] No secrets, weights, raw data, or `.env` committed.
- [ ] Issue/PR templates, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE` present.
- [ ] Repo description + topics set (e.g. `llm`, `qlora`, `mistral`, `fastapi`,
      `vllm`, `nextjs`, `fintech`, `mlops`).
- [ ] Pin the repo on your GitHub profile.

## Hugging Face — model publishing

1. Run real QLoRA training (GPU) and merge the adapter.
2. `huggingface-cli login`; create a model repo.
3. Upload the adapter (and/or merged model) with `scripts/upload_to_hub.py`.
4. Use [docs/model_card.md](model_card.md) as the `README.md` of the model repo;
   **fill in real evaluation numbers** before removing the “template / pending”
   banner.
5. Link the model from the GitHub README.

## Hugging Face — dataset publishing

1. Confirm SEC source terms and attribution (public-domain U.S. government data;
   respect EDGAR fair-access / rate limits).
2. Create a dataset repo; upload the processed instruction splits **only if** you
   are comfortable redistributing derived text.
3. Use [docs/dataset_card.md](dataset_card.md) as the dataset card.

## Demo deployment

- **Hugging Face Spaces** (frontend in demo mode) for a zero-GPU public demo.
- **Hosted GPU** (RunPod/Lambda) for the full stack — keep vLLM internal; expose
  only the FastAPI wrapper with a real `API_SECRET_KEY`.

## LinkedIn launch checklist

- [ ] Use **Version A** of [reports/linkedin_post.md](../reports/linkedin_post.md)
      until real training is done (no performance claims).
- [ ] Attach a short demo clip (see [reports/demo_script.md](../reports/demo_script.md)).
- [ ] Link the GitHub repo; tag relevant skills.

## Resume usage

- Use the “before training” bullets in
  [reports/resume_bullets.md](../reports/resume_bullets.md) now; swap in the
  “after training” bullets with real numbers once available.

## What NOT to claim before a real benchmark run

- ❌ “Improved accuracy/faithfulness by X%” — no real numbers exist yet.
- ❌ “Outperforms the base model” — not measured on a real adapter.
- ✅ “Built and tested the full fine-tuning + serving pipeline; real training
  pending GPU execution.”
