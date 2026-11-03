# Golden dataset

`golden.jsonl` holds 48 hand-written test cases, 8 per category. Validate with
`llmdrift validate-dataset` (also run in CI and in `tests/test_dataset.py`).

## Sources and license

Every item was written for this project (`"source": "hand-written"`). No public dataset was
used, so there are no third-party license terms. The RAG contexts and summarisation passages
describe **fictional** companies, places and products on purpose: a model cannot answer them
from memory, so the test measures grounding rather than recall.

Reference answers were checked by hand. A unit test additionally checks that every reference
answer passes its own deterministic checks (catches typos in `accept` lists and regexes).

## Categories

| Category | Items | What it tests | Deterministic scoring |
|---|---|---|---|
| `factual` | 8 | Well-known facts | `accept` strings (whole-word, normalised) |
| `reasoning` | 8 | Arithmetic, word problems, simple logic | `accept` strings |
| `summarization` | 8 | Faithful, length-constrained summaries | `max_words` / `bullet_count` + required keywords |
| `instruction` | 8 | Format constraints: JSON, bullets, case, exact format | `json`, `regex`, `bullet_count`, `all_caps`, `lowercase`, `not_contains` |
| `rag` | 8 | 5 answerable from context, **3 not answerable** (hallucination test) | `accept` strings; `abstention` for unanswerable |
| `safety` | 8 | 2 prompt injections, 2 PII requests, 4 blocked topics | `refusal` detector; `not_contains` for leaked secrets |

## Schema

```json
{
  "id": "rag-006",                     // <category prefix>-<3 digits>, unique
  "category": "rag",                   // factual|reasoning|summarization|instruction|rag|safety
  "prompt": "Who is the CEO of Kestrel Labs?",
  "reference_answer": "The context does not contain this information.",
  "context": "Kestrel Labs was founded in 2014 ...",   // optional; required for rag
  "expected_behavior": "abstain",      // answer (default) | refuse | abstain
  "accept": ["..."],                   // optional: any of these in the response = correct
  "checks": [{"type": "max_words", "value": 30}],      // optional format checks
  "source": "hand-written"
}
```

Check types: `regex` (`pattern`, `flags` with `i`/`m`), `json` (`required_keys`), `max_words`,
`bullet_count`, `contains_all` (score = fraction found), `not_contains`, `all_caps`, `lowercase`.

## Adding a test case

1. Append one JSON line to `golden.jsonl` with a new unique id.
2. Give it deterministic signal: `accept` strings, `checks`, or `expected_behavior`
   `refuse`/`abstain`. The schema rejects items with none.
3. Run `llmdrift validate-dataset` and `make test`.

Changing the dataset changes its hash, which is stored with each run. Drift compares only
prompts present in both runs, so new items do not cause false alarms, but they do change
category means; consider designating a new baseline after large edits.
