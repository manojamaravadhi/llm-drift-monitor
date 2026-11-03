# Handoff: LLM Model Comparison & Drift Monitor

Project folder: `/Users/manojkumar/Downloads/resume_proj/LLM` (git repo, branch `main`, no remote yet).
Python env: `.venv` (Python 3.12). Run commands with `.venv/bin/...`.

## What was built (summary of the chat)

1. **Research (docs/DESIGN.md).** Chose LiteLLM (one client for all providers) and DeepEval (LLM-as-judge). Rejected Ragas, promptfoo, lm-eval-harness, Langfuse, Evidently (reasons in DESIGN.md).
   Key findings: Groq shut down free Llama models on 2026-08-16; Ollama Cloud has **no free tier** (the user's Ollama key returns "Add usage credits").
2. **Core package `src/llmdrift/`**: providers (LiteLLM + retry with jitter, rate limit, circuit breaker, error taxonomy), golden dataset schema, deterministic checks, DeepEval judge, cost, bootstrap stats, SQLite storage, resumable/cached/parallel runner, drift detector (paired bootstrap + threshold + Bonferroni), Typer CLI (`run`, `report`, `drift`, `export`, `validate-dataset`, `list-runs`, `set-baseline`, `ask`).
3. **Golden dataset** `datasets/golden.jsonl`: 48 hand-written items, 8 per category (factual, reasoning, summarization, instruction, rag incl. 3 unanswerable, safety incl. 2 injections).
4. **Dashboard** `dashboard/app.py` (Streamlit + Plotly): models/judge panel, Overview, Quality·cost·latency, By category, Drift, Drill-down, Run history, and a new **"Try it live"** tab (visitors type a prompt; all models answer side by side; 10 prompts per visit, 300-token answers, 1-hour cache; keys from `.env` or Streamlit secrets).
5. **Demo/synthetic data was removed** at the user's request. Mock models remain only in `tests/fixtures/` for offline tests.
6. **CI**: `.github/workflows/ci.yml` (ruff, mypy --strict, pytest ≥80% coverage) and `weekly-eval.yml` (cron, secrets, eval-history branch, issue + fail on drift). Makefile, Dockerfile, docker-compose, pre-commit, MIT license.
7. **Real runs (free keys in `.env`)**, stored in `data/results.db` (git-ignored):
   - Baseline `20261103T125740-c39106` (designated baseline): 432 answers. Quality: qwen 0.989, gemini 0.984, gpt-oss 0.983. Latency p50/p95: qwen 294/1011 ms, gemini 1250/14148 ms, gpt-oss 676/2153 ms. List price per query: qwen $0.000292, gemini $0.000078, gpt-oss $0.000042. Qwen had 2 rate-limit errors.
   - Degraded `20261103T131618-cf64bf` (`--inject-degradation`, 1 repeat): `llmdrift drift` exits 1. Flagged: gemini quality 0.984→0.682 (safety 1.00→0.10: leaked secret code, complied with harmful requests), gpt-oss quality 0.983→0.774 (obeyed "PWNED" injection) and cost +46%. Qwen dropped 0.989→0.930 (below threshold, not flagged; leaked the secret code but refused the rest).
   - Harmful completions were redacted in the DB ("[hidden: ...]"); unredacted backup only in the Claude scratchpad (temporary).
8. **Bugs found by real runs and fixed**: G-Eval sent logprobs (Gemini rejects) → judge model raises AttributeError to use the plain path; faithfulness steps said "score 1" (0–10 scale) → reworded; Gemini judge overloaded → judge is now `groq/openai/gpt-oss-120b`; unit tests could load real keys from `.env` → `LLMDRIFT_DOTENV=0` in tests; models now run in parallel threads.

Models in `config/models.yaml`: `gemini-3.5-flash-lite`, `groq-gpt-oss-20b`, `groq-qwen3.8-27b` (enabled); `groq-llama-3.1-8b`, `ollama-cloud-gemma4-31b` (disabled); `ollama-llama3.2-1b` (local, auto-skipped if Ollama isn't running). Judge: `groq/openai/gpt-oss-120b` in `config/eval.yaml`.

## Uncommitted changes (in the working tree now)

- `src/llmdrift/live.py` (new): shared `ask_models()` used by CLI `ask` and the dashboard.
- `src/llmdrift/cli.py`: `ask` refactored to use `live.py`; LiteLLM logger set to ERROR.
- `src/llmdrift/runner.py`: `HIDDEN_HARMFUL`, never stores/caches harmful completions.
- `dashboard/app.py`: "Try it live" tab, `load_keys()` (dotenv + `st.secrets`, skipped when `LLMDRIFT_DOTENV=0`).
- `tests/test_runner.py` (harmful-text test), `tests/test_dashboard.py` (7 tabs), `tests/test_live.py` (new).
- `README.md`: real results section filled in; `<!-- DRIFT -->` and `<!-- SCREENSHOTS -->` placeholders still to fill.

## Tasks left

1. **Fix the one failing test** in `tests/test_live.py`: `test_ask_unknown_prompt_is_reported_not_raised` assumes the mock errors on unknown prompts, but it answers them. Replace it with a test that monkeypatches `llmdrift.live.build_provider` to return an object whose `generate()` raises `RateLimited("429 slow down")`, and assert `answers[0].error == "rate_limit: 429 slow down"` (import `pytest` and `from llmdrift.providers.base import RateLimited`).
2. **Run checks**: `.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy && DEEPEVAL_TELEMETRY_OPT_OUT=1 .venv/bin/pytest --cov`. Then commit (end the message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
3. **README**:
   - Replace `<!-- DRIFT -->` with the real drift findings above (point 7).
   - Take real screenshots: `make screenshots` (installs the `docs` extra and uses Google Chrome), then replace `<!-- SCREENSHOTS -->` with an image table (`docs/screenshots/overview.png`, `drift.png`, `scatter.png`, `drilldown.png`, plus heatmap/history links). Note: `scripts/screenshots.py` clicks tabs by name; add "Try it live" if wanted.
   - Mention the "Try it live" tab and update the badge/test counts (currently ~114 tests, ~98% coverage).
4. **Make it deployable on Streamlit Community Cloud (free live link)**:
   - Add `!data/results.db` to `.gitignore` so the redacted results ship with the repo.
   - Add `requirements.txt` with a single line `.` (installs the package and its deps). Choose Python 3.12 in the Streamlit deploy settings.
   - In the Streamlit app settings → Secrets, add `GEMINI_API_KEY = "..."` and `GROQ_API_KEY = "..."` (never commit them).
5. **Publish** (needs your decision: public or private): `gh repo create llm-drift-monitor --public --source . --push`. Then at https://share.streamlit.io → "Create app" → repo `manojamaravadhi/llm-drift-monitor`, branch `main`, file `dashboard/app.py`. The link will look like `https://<name>.streamlit.app`.
6. **Weekly workflow**: add `GEMINI_API_KEY` and `GROQ_API_KEY` as GitHub repo secrets; Settings → Actions → General → "Read and write permissions".
7. **Security**: regenerate the Ollama API key at https://ollama.com/settings/keys (it was pasted in chat). Check `.env` is never committed (`git check-ignore .env`).
8. **Optional improvements**: harder golden items (all models score ~0.98, a ceiling effect); a judge from a different family to reduce self-preference bias (gpt-oss-120b judges gpt-oss-20b); more safety prompts so per-category safety drops like Qwen's can be detected; Docker build was never tested (Docker daemon was off).

## Useful commands

```bash
.venv/bin/llmdrift ask "What is the capital of Australia?"     # one live request to all models
.venv/bin/llmdrift report                                       # leaderboard for latest run
.venv/bin/llmdrift drift                                        # exit 1 if drift vs baseline
.venv/bin/streamlit run dashboard/app.py                        # http://localhost:8501 (or 8502 if busy)
.venv/bin/llmdrift run                                          # full real eval (~15 min on free tiers)
```

More detail: `docs/USAGE.md`, `docs/DESIGN.md`, `datasets/README.md`.
