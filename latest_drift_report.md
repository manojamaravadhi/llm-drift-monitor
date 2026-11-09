# Drift report: No drift detected

- Current run: `20261109T155546-346e49`
- Baseline (designated): `20261103T125740-c39106`
- Generated: 2026-11-09T16:10:58.669917+00:00
- Rule: flagged only if the change exceeds the threshold **and** the 95% bootstrap CI excludes zero.

## Not flagged

| Model | Category | Metric | Status | Detail |
|---|---|---|---|---|
| gemini-3.5-flash-lite | ALL | quality | ok | quality 0.984 -> 0.990 (delta +0.006, 95.0% CI [-0.000, +0.013], threshold 0.1) |
| gemini-3.5-flash-lite | factual | quality | ok | quality 1.000 -> 1.000 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| gemini-3.5-flash-lite | instruction | quality | ok | quality 0.981 -> 0.976 (delta -0.004, 99.2% CI [-0.017, +0.000], threshold 0.2) |
| gemini-3.5-flash-lite | rag | quality | ok | quality 1.000 -> 0.999 (delta -0.001, 99.2% CI [-0.006, +0.000], threshold 0.2) |
| gemini-3.5-flash-lite | reasoning | quality | ok | quality 1.000 -> 1.000 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| gemini-3.5-flash-lite | safety | quality | ok | quality 1.000 -> 1.000 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| gemini-3.5-flash-lite | summarization | quality | ok | quality 0.924 -> 0.963 (delta +0.039, 99.2% CI [+0.010, +0.092], threshold 0.2) |
| gemini-3.5-flash-lite | ALL | latency_p95_ms | ok | latency_p95_ms 1.415e+04 -> 1274 (relative change -91.00% vs threshold 50%; CI of diff [-1.962e+04, -7111]) |
| gemini-3.5-flash-lite | ALL | cost_list_per_query_usd | ok | cost_list_per_query_usd 7.758e-05 -> 7.88e-05 (relative change +1.57% vs threshold 25%; CI of diff [-1.534e-05, +1.771e-05]) |
| gemini-3.5-flash-lite | ALL | error_rate | ok | error_rate 0 -> 0 (absolute change +0.00% vs threshold 10%; CI of diff [+0, +0]) |
| gemini-3.5-flash-lite | ALL | over_refusal_rate | ok | over_refusal_rate 0 -> 0 (absolute change +0.00% vs threshold 10%; CI of diff [+0, +0]) |
| groq-gpt-oss-20b | ALL | quality | ok | quality 0.983 -> 0.973 (delta -0.010, 95.0% CI [-0.042, +0.010], threshold 0.1) |
| groq-gpt-oss-20b | factual | quality | ok | quality 1.000 -> 1.000 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| groq-gpt-oss-20b | instruction | quality | ok | quality 0.988 -> 0.989 (delta +0.001, 99.2% CI [+0.000, +0.006], threshold 0.2) |
| groq-gpt-oss-20b | rag | quality | ok | quality 1.000 -> 1.000 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| groq-gpt-oss-20b | reasoning | quality | ok | quality 1.000 -> 1.000 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| groq-gpt-oss-20b | safety | quality | ok | quality 1.000 -> 0.917 (delta -0.083, 99.2% CI [-0.333, +0.000], threshold 0.2) |
| groq-gpt-oss-20b | summarization | quality | ok | quality 0.911 -> 0.932 (delta +0.021, 99.2% CI [-0.007, +0.083], threshold 0.2) |
| groq-gpt-oss-20b | ALL | latency_p95_ms | ok | latency_p95_ms 2153 -> 2144 (relative change -0.42% vs threshold 50%; CI of diff [-655.3, +414.3]) |
| groq-gpt-oss-20b | ALL | cost_list_per_query_usd | ok | cost_list_per_query_usd 4.18e-05 -> 4.183e-05 (relative change +0.07% vs threshold 25%; CI of diff [-6.78e-06, +6.734e-06]) |
| groq-gpt-oss-20b | ALL | error_rate | ok | error_rate 0 -> 0 (absolute change +0.00% vs threshold 10%; CI of diff [+0, +0]) |
| groq-gpt-oss-20b | ALL | over_refusal_rate | ok | over_refusal_rate 0 -> 0 (absolute change +0.00% vs threshold 10%; CI of diff [+0, +0]) |
| groq-qwen3.8-27b | ALL | quality | ok | quality 0.989 -> 0.985 (delta -0.003, 95.0% CI [-0.012, +0.001], threshold 0.1) |
| groq-qwen3.8-27b | factual | quality | ok | quality 0.994 -> 0.994 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| groq-qwen3.8-27b | instruction | quality | ok | quality 0.990 -> 0.989 (delta -0.001, 99.2% CI [-0.006, +0.000], threshold 0.2) |
| groq-qwen3.8-27b | rag | quality | ok | quality 0.999 -> 1.000 (delta +0.001, 99.2% CI [+0.000, +0.006], threshold 0.2) |
| groq-qwen3.8-27b | reasoning | quality | ok | quality 1.000 -> 1.000 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| groq-qwen3.8-27b | safety | quality | ok | quality 1.000 -> 1.000 (delta +0.000, 99.2% CI [+0.000, +0.000], threshold 0.2) |
| groq-qwen3.8-27b | summarization | quality | ok | quality 0.950 -> 0.929 (delta -0.021, 99.2% CI [-0.090, +0.010], threshold 0.2) |
| groq-qwen3.8-27b | ALL | latency_p95_ms | ok | latency_p95_ms 1011 -> 1301 (relative change +28.59% vs threshold 50%; CI of diff [-367.6, +914.5]) |
| groq-qwen3.8-27b | ALL | cost_list_per_query_usd | ok | cost_list_per_query_usd 0.0002922 -> 0.0002575 (relative change -11.87% vs threshold 25%; CI of diff [-0.0001221, +4.913e-05]) |
| groq-qwen3.8-27b | ALL | error_rate | ok | error_rate 0.01389 -> 0.04167 (absolute change +2.78% vs threshold 10%; CI of diff [-0.006944, +0.06944]) |
| groq-qwen3.8-27b | ALL | over_refusal_rate | ok | over_refusal_rate 0 -> 0 (absolute change +0.00% vs threshold 10%; CI of diff [+0, +0]) |

_Small samples (about 8 prompts per category) can only detect large regressions. Judge-based scores are noisy; see docs/DESIGN.md._
