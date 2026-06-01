# FinSage-7B — FAQ

### Is this financial advice?

No. FinSage-7B is **not** a licensed financial advisor. Outputs are informational
summaries of supplied text only, are not investment recommendations, and may be
incomplete or wrong. Always verify against the original filing.

### Is the model actually trained?

The full training **pipeline** is implemented and tested in CPU dry-run mode, but
the **real GPU training run is pending** (RunPod/A100). Any numbers in the
benchmark report today are sample/pipeline-validation results from a mock backend,
clearly labelled as such — not real model performance.

### Why SEC filings?

They are public, well-structured, long, dense, and high-stakes — a credible
domain to demonstrate that a small fine-tuned model can be made more grounded and
specific than its base model on a real professional task.

### Why not use only RAG?

RAG and fine-tuning are complementary. Fine-tuning teaches *format, task
behaviour, and grounding discipline* (e.g. avoiding advice, extracting numbers
faithfully) that retrieval alone does not. A RAG baseline is on the roadmap for
comparison; in production you would likely combine both (retrieve relevant filing
sections, then let a task-tuned model answer).

### Why QLoRA?

It makes 7B fine-tuning affordable on a single GPU: 4-bit quantisation cuts memory
while LoRA adapters keep trainable parameters tiny. The resulting adapter is small
and easy to share/merge.

### Can this be used for Indian filings (or other jurisdictions)?

Not out of the box — the ingestion and section extraction target SEC EDGAR and
U.S. filing structure. The pipeline is adaptable (swap the ingestion + section
heuristics and rebuild the dataset), but results on non-SEC filings are untested.

### What are the main limitations?

GPU-dependent training (pending), weak-supervision labels, lexical (not NLI)
faithfulness by default, a small evaluation set, demo-grade API-key auth, and an
in-memory rate limiter. See the README Limitations section.

### How would you productionize this further?

Run real training and publish real numbers; upgrade labels and add NLI / calibrated
LLM-as-judge evaluation; switch to Redis-backed rate limiting and gateway/OAuth
auth; add tracing, dashboards, and alerting; autoscale vLLM with canary model
rollouts; keep vLLM internal behind the authenticated FastAPI wrapper.
