"""Load results from SQLite into pandas and compute leaderboard statistics.

Shared by the CLI (``report``/``export``), the drift detector and the dashboard, so every
surface shows the same numbers.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from llmdrift.metrics.stats import percentile
from llmdrift.storage.db import MetricRow, Result, Run

RESULT_COLUMNS = [
    "run_id",
    "model_name",
    "model_id",
    "prompt_id",
    "category",
    "expected_behavior",
    "repeat_idx",
    "response_text",
    "error_type",
    "error_message",
    "latency_ms",
    "ttft_ms",
    "input_tokens",
    "output_tokens",
    "tokens_estimated",
    "cost_billed_usd",
    "cost_list_usd",
    "quality_score",
    "refused",
    "from_cache",
]


def load_runs(engine: Engine) -> pd.DataFrame:
    """All runs, oldest first."""
    stmt = select(
        Run.run_id,
        Run.started_at,
        Run.finished_at,
        Run.status,
        Run.git_commit,
        Run.label,
        Run.is_baseline,
        Run.synthetic,
        Run.config_json,
    ).order_by(Run.started_at)
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(stmt).mappings().all())
    if df.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "started_at",
                "finished_at",
                "status",
                "git_commit",
                "label",
                "is_baseline",
                "synthetic",
                "config_json",
            ]
        )
    return df


def load_results(engine: Engine, run_ids: list[str] | None = None) -> pd.DataFrame:
    """Result rows joined with run start time."""
    cols = [getattr(Result, c) for c in RESULT_COLUMNS] + [Result.id.label("result_id")]
    stmt = select(*cols, Run.started_at).join(Run, Run.run_id == Result.run_id)
    if run_ids is not None:
        stmt = stmt.where(Result.run_id.in_(run_ids))
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(stmt).mappings().all())
    if df.empty:
        return pd.DataFrame(columns=[*RESULT_COLUMNS, "result_id", "started_at"])
    return df


def load_metrics(engine: Engine, run_ids: list[str] | None = None) -> pd.DataFrame:
    """Metric rows with their result's run, model and prompt."""
    stmt = select(
        MetricRow.result_id,
        MetricRow.name,
        MetricRow.value,
        MetricRow.source,
        MetricRow.reason,
        Result.run_id,
        Result.model_name,
        Result.prompt_id,
        Result.category,
        Result.repeat_idx,
    ).join(Result, Result.id == MetricRow.result_id)
    if run_ids is not None:
        stmt = stmt.where(Result.run_id.in_(run_ids))
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(stmt).mappings().all())
    if df.empty:
        return pd.DataFrame(
            columns=[
                "result_id",
                "name",
                "value",
                "source",
                "reason",
                "run_id",
                "model_name",
                "prompt_id",
                "category",
                "repeat_idx",
            ]
        )
    return df


def per_prompt_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of quality across repeats, per (run, model, prompt)."""
    ok = results[results["error_type"].isna() & results["quality_score"].notna()]
    grouped = ok.groupby(["run_id", "model_name", "category", "prompt_id"])["quality_score"]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={"mean": "score", "std": "score_std", "count": "n_repeats"})


def hallucination_flags(results: pd.DataFrame, metrics: pd.DataFrame) -> pd.Series:
    """Boolean per result: abstain item answered anyway, or judge faithfulness < 0.5."""
    if metrics.empty or results.empty:
        return pd.Series(False, index=results.index)
    bad = metrics[
        ((metrics["name"] == "abstention") & (metrics["value"] < 0.5))
        | ((metrics["name"] == "judge_faithfulness") & (metrics["value"] < 0.5))
    ]["result_id"]
    return results["result_id"].isin(set(bad))


def leaderboard(results: pd.DataFrame, metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per model with quality, latency, cost, error and refusal statistics."""
    if results.empty:
        return pd.DataFrame()
    prompt_scores = per_prompt_scores(results)
    is_halluc_item = (results["expected_behavior"] == "abstain") | (results["category"] == "rag")
    halluc = (
        hallucination_flags(results, metrics)
        if metrics is not None
        else pd.Series(False, index=results.index)
    )
    rows = []
    for model, grp in results.groupby("model_name"):
        ok = grp[grp["error_type"].isna()]
        ps = prompt_scores[prompt_scores["model_name"] == model]
        answer_items = ok[ok["expected_behavior"] == "answer"]
        h_items = ok[is_halluc_item.loc[ok.index]]
        list_costs = ok["cost_list_usd"].dropna()
        rows.append(
            {
                "model": model,
                "quality": ps["score"].mean() if not ps.empty else None,
                "quality_std_across_repeats": ps["score_std"].mean() if not ps.empty else None,
                "latency_p50_ms": percentile(ok["latency_ms"].dropna().tolist(), 50),
                "latency_p95_ms": percentile(ok["latency_ms"].dropna().tolist(), 95),
                "ttft_p50_ms": percentile(ok["ttft_ms"].dropna().tolist(), 50),
                "cost_billed_per_query_usd": ok["cost_billed_usd"].mean() if len(ok) else None,
                "cost_list_per_query_usd": list_costs.mean() if len(list_costs) else None,
                "cost_list_total_usd": list_costs.sum() if len(list_costs) else None,
                "error_rate": grp["error_type"].notna().mean(),
                "over_refusal_rate": answer_items["refused"].mean() if len(answer_items) else None,
                "hallucination_rate": halluc.loc[h_items.index].mean() if len(h_items) else None,
                "tokens_estimated_share": ok["tokens_estimated"].mean() if len(ok) else None,
                "n_responses": len(grp),
                "n_prompts": grp["prompt_id"].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values("quality", ascending=False, na_position="last")


def category_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Mean per-prompt quality, models as rows and categories as columns."""
    ps = per_prompt_scores(results)
    if ps.empty:
        return pd.DataFrame()
    return ps.pivot_table(index="model_name", columns="category", values="score", aggfunc="mean")


def run_timeseries(results: pd.DataFrame) -> pd.DataFrame:
    """Mean quality, p95 latency and error rate per (run, model) over time."""
    if results.empty:
        return pd.DataFrame()
    ps = per_prompt_scores(results)
    quality = ps.groupby(["run_id", "model_name"])["score"].mean().rename("quality")
    meta = results.groupby(["run_id", "model_name"]).agg(
        started_at=("started_at", "first"),
        error_rate=("error_type", lambda s: s.notna().mean()),
        latency_p95_ms=("latency_ms", lambda s: percentile(s.dropna().tolist(), 95)),
        cost_list_per_query_usd=("cost_list_usd", "mean"),
    )
    return meta.join(quality).reset_index().sort_values("started_at")
