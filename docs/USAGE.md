# Using it by hand

All commands run from the project folder. `make setup` installs everything into `.venv`; the commands below use `.venv/bin/llmdrift` (or run `source .venv/bin/activate` once and just type `llmdrift`).

## 1. Keys

Put free keys in `.env` (git-ignored):

```
GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
GROQ_API_KEY=...     # https://console.groq.com/keys
```

Models without a key are skipped with a message, so one key is enough to start.

## 2. Ask one question (request → responses)

```bash
.venv/bin/llmdrift ask "What is the capital of Australia?"
.venv/bin/llmdrift ask "Who founded the company?" --context "Acme was founded by Ada Lee in 2019."
.venv/bin/llmdrift ask "Explain drift in one sentence" --models groq-gpt-oss-20b
```

Each model prints its response, latency, time to first token, tokens and cost. Nothing is stored. This is the quickest way to see what each model does with a request.

**What is sent (the request).** For every model the same chat request goes through LiteLLM:

| Part | Value (from `config/eval.yaml`) |
|---|---|
| System message | "You are a helpful, accurate assistant. Answer concisely." |
| User message | Your prompt. With `--context` (or a RAG item): "Answer the question using only the context below. If the context does not contain the answer, say that the context does not contain it. Context: … Question: …" |
| Parameters | temperature 0.2, max 512 tokens, streaming on (to measure time to first token) |

## 3. Full evaluation (48 prompts × 3 repeats × every model)

```bash
.venv/bin/llmdrift run                     # about 15 min on free tiers; resumes if interrupted
.venv/bin/llmdrift run --models groq-gpt-oss-20b --categories safety,rag --runs 1   # smaller
.venv/bin/llmdrift report                  # leaderboard in the terminal + reports/leaderboard.md
.venv/bin/llmdrift list-runs
```

Every response is stored in `data/results.db` (SQLite) with its score, latency, tokens, cost and any error.

## 4. Read the stored responses

Dashboard (easiest):

```bash
.venv/bin/streamlit run dashboard/app.py
```

Then open **http://localhost:8501** (if that port is busy, Streamlit prints the one it used, e.g. 8502). Tabs:

| Tab | Shows |
|---|---|
| Header panel | Models (provider, model id) and the judge used for the selected run |
| Overview | Leaderboard: quality, latency p50/p95, time to first token, cost, error / over-refusal / hallucination rates |
| Quality · cost · latency | Trade-off scatter |
| By category | Model × category heatmap |
| Drift | Quality over runs with the baseline band; flagged runs marked; full drift report |
| Drill-down | Pick a prompt: every model's response side by side with each metric, the judge's reason, latency and tokens. "Failures only" narrows to problems |
| Run history | All runs and each run's saved config |

Files:

```bash
.venv/bin/llmdrift export --format csv --out reports/results.csv     # one row per response, metrics as columns
.venv/bin/llmdrift export --format jsonl --out reports/results.jsonl
```

SQL:

```bash
sqlite3 data/results.db "select model_name, prompt_id, quality_score, response_text from results limit 10;"
sqlite3 data/results.db "select r.model_name, r.prompt_id, m.name, m.value, m.reason from metrics m join results r on r.id = m.result_id where m.source = 'judge' limit 10;"
```

Tables: `runs` (one per evaluation, with config snapshot and git commit), `results` (one per response), `metrics` (one per score, with the judge's reason), `response_cache`.

## 5. Drift

```bash
.venv/bin/llmdrift drift                    # latest run vs. baseline; exit 1 if drift
.venv/bin/llmdrift run --fresh --no-cache --runs 1 --inject-degradation   # deliberately worse run
.venv/bin/llmdrift drift                    # now flags the regression
.venv/bin/llmdrift set-baseline <run_id>    # choose which run counts as "normal"
```

Reports land in `reports/drift_report.md` and `reports/drift_report.json`.

## 6. Change things

| To | Edit |
|---|---|
| Add or remove a model | `config/models.yaml` (any LiteLLM model or OpenAI-compatible endpoint) |
| Change prices | `config/pricing.yaml` |
| Change repeats, temperature, system prompt, judge, drift thresholds | `config/eval.yaml` |
| Add test prompts | `datasets/golden.jsonl`, then `llmdrift validate-dataset` |
