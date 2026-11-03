"""Streamlit dashboard. Reads SQLite only; never calls a model API.

Run: ``streamlit run dashboard/app.py`` (set LLMDRIFT_DB to choose the database).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from llmdrift.analysis import (
    category_matrix,
    leaderboard,
    load_metrics,
    load_results,
    load_runs,
    run_timeseries,
)
from llmdrift.config import DriftConfig, load_eval_config
from llmdrift.drift.baseline import BaselineError, choose_baseline
from llmdrift.drift.detector import DriftReport, detect_drift
from llmdrift.drift.history import drift_history
from llmdrift.storage.db import get_engine

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (validated): categorical slots in fixed order, reserved status colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="LLM Drift Monitor", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
      .badge {display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.8rem;
              background:#f3f2ee;color:#52514e;margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data
def default_db() -> str:
    """LLMDRIFT_DB, else the real results DB if it exists, else the bundled demo DB."""
    if os.environ.get("LLMDRIFT_DB"):
        return os.environ["LLMDRIFT_DB"]
    real = ROOT / "data" / "results.db"
    return str(real if real.exists() else ROOT / "data" / "demo.db")


def _mtime(path: str) -> float:
    return Path(path).stat().st_mtime if Path(path).exists() else 0.0


@st.cache_data(show_spinner=False)
def get_data(db_path: str, mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs, results and metrics. ``mtime`` busts the cache when the DB changes."""
    engine = get_engine(db_path, create=False)
    try:
        return load_runs(engine), load_results(engine), load_metrics(engine)
    finally:
        engine.dispose()


@st.cache_data(show_spinner=False)
def get_drift(
    db_path: str, mtime: float, run_id: str, cfg_json: str
) -> tuple[DriftReport | None, str]:
    """Drift report for one run against its configured baseline."""
    cfg = DriftConfig.model_validate_json(cfg_json)
    engine = get_engine(db_path, create=False)
    try:
        choice = choose_baseline(engine, cfg, current_run=run_id)
        report = detect_drift(
            load_results(engine, [choice.current_run]),
            load_results(engine, choice.baseline_runs),
            cfg,
            choice.current_run,
            choice.baseline_runs,
            choice.mode,
        )
        return report, ""
    except BaselineError as exc:
        return None, str(exc)
    finally:
        engine.dispose()


@st.cache_data(show_spinner=False)
def get_history(db_path: str, mtime: float, cfg_json: str) -> pd.DataFrame:
    """Per-run rolling drift status for the time series."""
    cfg = DriftConfig.model_validate_json(cfg_json).model_copy(update={"bootstrap_samples": 500})
    runs, results, _ = get_data(db_path, mtime)
    return drift_history(results, runs, cfg)


def color_map(models: list[str]) -> dict[str, str]:
    """Fixed colour per model (by sorted name over the whole DB, so filters never repaint)."""
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(sorted(models))}


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    """Recessive axes and grid, consistent fonts."""
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK_2, "size": 13},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "y": 1.08, "x": 0, "title": None},
        hoverlabel={"bgcolor": "white", "font_color": INK},
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont_color=MUTED)
    return fig


def fmt_money(value: Any) -> str:
    """Small USD amounts, or n/a."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("LLM Drift Monitor")
db_path = st.sidebar.text_input("SQLite database", value=default_db())
if not Path(db_path).exists():
    st.title("LLM Model Comparison & Drift Monitor")
    st.info(
        f"No database at `{db_path}`.\n\n"
        "- Demo data (no API keys): `make demo`\n"
        "- Real evaluation: add keys to `.env`, then `make run`"
    )
    st.stop()

mtime = _mtime(db_path)
runs, results, metrics = get_data(db_path, mtime)
completed = runs[runs["status"] == "completed"]
if completed.empty or results.empty:
    st.title("LLM Model Comparison & Drift Monitor")
    st.info(
        "The database has no completed runs yet. Run `make run` (needs API keys) or "
        "`make demo` to load the synthetic demo history."
    )
    st.stop()

eval_cfg = load_eval_config(ROOT / "config" / "eval.yaml")
drift_cfg_json = eval_cfg.drift.model_dump_json()
all_models = sorted(results["model_name"].unique())
COLORS = color_map(all_models)

run_ids = completed["run_id"].tolist()[::-1]
run_labels: dict[str, str] = {
    str(rid): f"{started:%Y-%m-%d} · {label or rid}"
    for rid, started, label in zip(
        completed["run_id"], completed["started_at"], completed["label"], strict=True
    )
}
run_id = st.sidebar.selectbox("Run", run_ids, format_func=lambda r: run_labels[r])
models_sel = st.sidebar.multiselect("Models", all_models, default=all_models)
categories = sorted(results["category"].unique())
cats_sel = st.sidebar.multiselect("Categories", categories, default=categories)
st.sidebar.caption(f"{len(completed)} completed runs · {len(all_models)} models")

run_row = completed.set_index("run_id").loc[run_id]
is_synthetic = bool(run_row["synthetic"])

view = results[
    (results["run_id"] == run_id)
    & results["model_name"].isin(models_sel)
    & results["category"].isin(cats_sel)
]
view_metrics = metrics[metrics["result_id"].isin(view["result_id"])]

# ----------------------------------------------------------------------------- header
st.title("LLM Model Comparison & Drift Monitor")
badges = [
    f"<span class='badge'>run {run_id}</span>",
    f"<span class='badge'>commit {run_row['git_commit'] or 'n/a'}</span>",
    f"<span class='badge'>{run_row['started_at']:%Y-%m-%d %H:%M} UTC</span>",
]
st.markdown(" ".join(badges), unsafe_allow_html=True)
if is_synthetic:
    st.warning(
        "**Synthetic demo data.** These runs were generated offline by mock models "
        "(`scripts/seed_demo_db.py`). The numbers are invented and say nothing about real "
        "Gemini, Groq or Ollama models.",
        icon="🧪",
    )

report, drift_err = get_drift(db_path, mtime, run_id, drift_cfg_json)
if report is not None and report.drift_detected:
    flagged_models = sorted({f.model for f in report.flagged})
    st.error(
        f"**Drift detected** in {len(report.flagged)} check(s) for "
        f"{', '.join(flagged_models)} (baseline: {report.baseline_mode}). See the Drift tab.",
        icon="🚨",
    )
elif report is not None:
    st.success(f"No drift vs baseline ({report.baseline_mode}).", icon="✅")

tabs = st.tabs(
    ["Overview", "Quality · cost · latency", "By category", "Drift", "Drill-down", "Run history"]
)

# ----------------------------------------------------------------------------- 1 overview
with tabs[0]:
    board = leaderboard(view, view_metrics)
    if board.empty:
        st.info("No results for this selection.")
    else:
        best = board.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top quality", f"{best['quality']:.3f}", best["model"], delta_color="off")
        fastest = board.sort_values("latency_p50_ms").iloc[0]
        c2.metric(
            "Fastest p50",
            f"{fastest['latency_p50_ms']:.0f} ms",
            fastest["model"],
            delta_color="off",
        )
        c3.metric("Responses", f"{int(board['n_responses'].sum()):,}")
        c4.metric("Billed cost (this run)", fmt_money(view["cost_billed_usd"].sum()))

        table = board.rename(
            columns={
                "model": "Model",
                "quality": "Quality",
                "quality_std_across_repeats": "Std across repeats",
                "latency_p50_ms": "Latency p50 (ms)",
                "latency_p95_ms": "Latency p95 (ms)",
                "ttft_p50_ms": "TTFT p50 (ms)",
                "cost_billed_per_query_usd": "Billed $/query",
                "cost_list_per_query_usd": "List-price $/query",
                "error_rate": "Error rate",
                "over_refusal_rate": "Over-refusal",
                "hallucination_rate": "Hallucination",
                "n_responses": "Responses",
            }
        )
        cols = [c for c in table.columns if not c.endswith("_usd") and not c.startswith("n_")]
        cols = [c for c in cols if c not in {"tokens_estimated_share"}]
        st.dataframe(
            table[cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Quality": st.column_config.ProgressColumn(
                    "Quality", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "Std across repeats": st.column_config.NumberColumn(format="%.3f"),
                "Latency p50 (ms)": st.column_config.NumberColumn(format="%.0f"),
                "Latency p95 (ms)": st.column_config.NumberColumn(format="%.0f"),
                "TTFT p50 (ms)": st.column_config.NumberColumn(format="%.0f"),
                "Billed $/query": st.column_config.NumberColumn(format="$%.6f"),
                "List-price $/query": st.column_config.NumberColumn(format="$%.6f"),
                "Error rate": st.column_config.NumberColumn(format="percent"),
                "Over-refusal": st.column_config.NumberColumn(format="percent"),
                "Hallucination": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "**Billed** = what you actually pay (free tiers and local models: $0). "
            "**List-price** = the same tokens at the paid list price in `config/pricing.yaml` "
            "(n/a for local models). Quality = mean of deterministic checks"
            " and, when enabled, LLM-judge scores; per prompt it is averaged over repeats."
        )
        if board["tokens_estimated_share"].fillna(0).gt(0).any():
            st.caption("Some token counts were estimated with a tokenizer (provider gave none).")

# ----------------------------------------------------------------------------- 2 scatter
with tabs[1]:
    board = leaderboard(view, view_metrics)
    if board.empty:
        st.info("No results for this selection.")
    else:
        latency_col = st.radio(
            "Latency",
            ["latency_p50_ms", "latency_p95_ms"],
            horizontal=True,
            format_func=lambda c: c.split("_")[1].upper(),
        )
        plot = board.copy()
        plot["cost_for_size"] = plot["cost_list_per_query_usd"].fillna(0)
        max_cost = plot["cost_for_size"].max() or 1.0
        plot["bubble"] = 14 + 46 * plot["cost_for_size"] / max_cost
        plot["cost_label"] = plot["cost_list_per_query_usd"].map(fmt_money)
        fig = go.Figure()
        for row in plot.itertuples():
            fig.add_trace(
                go.Scatter(
                    x=[getattr(row, latency_col)],
                    y=[row.quality],
                    mode="markers+text",
                    name=row.model,
                    text=[row.model],
                    textposition="top center",
                    textfont={"color": INK_2},
                    marker={
                        "size": row.bubble,
                        "color": COLORS[str(row.model)],
                        "opacity": 0.85,
                        "line": {"color": SURFACE, "width": 2},
                    },
                    hovertemplate=(
                        f"<b>{row.model}</b><br>quality %{{y:.3f}}<br>"
                        f"latency %{{x:.0f}} ms<br>list-price/query {row.cost_label}"
                        "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            xaxis_title=f"Latency {latency_col.split('_')[1].upper()} (ms) · lower is better",
            yaxis_title="Quality · higher is better",
            yaxis_range=[0, 1.1],
            xaxis_range=[0, float(plot[latency_col].max()) * 1.25],
        )
        st.plotly_chart(style(fig, 480), use_container_width=True)
        st.caption(
            "Bubble size = list-price cost per query. Models with no list price (local) are "
            "drawn at the minimum size. Top-left is best."
        )

# ----------------------------------------------------------------------------- 3 heatmap
with tabs[2]:
    matrix = category_matrix(view)
    if matrix.empty:
        st.info("No results for this selection.")
    else:
        fig = px.imshow(
            matrix,
            text_auto=".2f",
            zmin=0,
            zmax=1,
            aspect="auto",
            color_continuous_scale=BLUE_RAMP,
            labels={"x": "Category", "y": "Model", "color": "Quality"},
        )
        fig.update_traces(xgap=2, ygap=2, hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>")
        st.plotly_chart(style(fig, 120 + 60 * len(matrix)), use_container_width=True)
        st.caption("Mean per-prompt quality (0–1). Each category has about 8 prompts.")

# ----------------------------------------------------------------------------- 4 drift
with tabs[3]:
    series = run_timeseries(results[results["model_name"].isin(models_sel)])
    history = get_history(db_path, mtime, drift_cfg_json)
    if series.empty:
        st.info("No runs yet.")
    else:
        models_in = [m for m in all_models if m in set(series["model_name"])]
        fig = make_subplots(
            rows=len(models_in),
            cols=1,
            shared_xaxes=True,
            subplot_titles=models_in,
            vertical_spacing=0.08,
        )
        flag_legend_shown = False
        for i, model in enumerate(models_in, start=1):
            s = series[series["model_name"] == model].merge(
                history, on=["run_id", "model_name"], how="left"
            )
            band = s.dropna(subset=["band_low"])
            if not band.empty:
                fig.add_trace(
                    go.Scatter(
                        x=list(band["started_at"]) + list(band["started_at"])[::-1],
                        y=list(band["band_high"].clip(upper=1))
                        + list(band["band_low"].clip(lower=0))[::-1],
                        fill="toself",
                        mode="lines",
                        fillcolor="rgba(137,135,129,0.15)",
                        line={"width": 0},
                        hoverinfo="skip",
                        name="baseline ± threshold",
                        showlegend=i == 1,
                    ),
                    row=i,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(
                    x=s["started_at"],
                    y=s["quality"],
                    mode="lines+markers",
                    line={"color": COLORS[model], "width": 2},
                    marker={
                        "size": 8,
                        "color": COLORS[model],
                        "line": {"color": SURFACE, "width": 2},
                    },
                    name=model,
                    showlegend=False,
                    customdata=s[["run_id", "latency_p95_ms", "error_rate"]].fillna(0),
                    hovertemplate=(
                        "%{customdata[0]}<br>quality %{y:.3f}<br>p95 %{customdata[1]:.0f} ms"
                        "<br>errors %{customdata[2]:.1%}<extra>" + model + "</extra>"
                    ),
                ),
                row=i,
                col=1,
            )
            flagged = s[s["flagged"].fillna("") != ""]
            if not flagged.empty:
                fig.add_trace(
                    go.Scatter(
                        x=flagged["started_at"],
                        y=flagged["quality"],
                        mode="markers+text",
                        marker={"symbol": "x", "size": 14, "color": CRITICAL},
                        text=["drift"] * len(flagged),
                        textposition="bottom center",
                        textfont={"color": CRITICAL},
                        name="drift flagged",
                        showlegend=not flag_legend_shown,
                        customdata=flagged[["flagged"]],
                        hovertemplate="flagged: %{customdata[0]}<extra></extra>",
                    ),
                    row=i,
                    col=1,
                )
                flag_legend_shown = True
            fig.update_yaxes(range=[0, 1.05], row=i, col=1)
        st.plotly_chart(style(fig, 230 * len(models_in) + 60), use_container_width=True)
        st.caption(
            "Each point is one run. The grey band is the rolling baseline "
            f"(median of the previous {eval_cfg.drift.rolling_k} runs) ± the quality threshold "
            f"({eval_cfg.drift.quality_threshold}). A red ✕ marks a run where drift was flagged: "
            "the drop exceeded the threshold and the bootstrap CI excluded zero."
        )

    st.subheader("Drift report for the selected run")
    if report is None:
        st.info(f"No drift report: {drift_err}")
    else:
        st.markdown(
            f"Baseline **{report.baseline_mode}**: "
            + ", ".join(f"`{r}`" for r in report.baseline_runs)
        )
        findings = pd.DataFrame([f.__dict__ for f in report.findings])
        findings = findings[findings["model"].isin(models_sel)]
        only_flagged = st.toggle("Flagged only", value=report.drift_detected)
        if only_flagged:
            findings = findings[findings["status"] == "drift"]
        st.dataframe(
            findings[["status", "model", "category", "metric", "message", "n"]],
            hide_index=True,
            use_container_width=True,
        )

# ----------------------------------------------------------------------------- 5 drill-down
with tabs[4]:
    if view.empty:
        st.info("No results for this selection.")
    else:
        failing = view[(view["quality_score"].fillna(0) < 1) | view["error_type"].notna()]
        only_fail = st.toggle(
            "Failures only",
            value=False,
            help="Prompts where any model scored below 1.0 or errored.",
        )
        prompt_ids = sorted((failing if only_fail else view)["prompt_id"].unique())
        if not prompt_ids:
            st.success("No failures in this selection.")
        else:
            from llmdrift.datasets.schema import load_dataset

            items = {i.id: i for i in load_dataset(ROOT / eval_cfg.dataset)}
            pid = st.selectbox(
                "Prompt",
                prompt_ids,
                format_func=lambda p: f"{p} · {items[p].prompt[:80]}" if p in items else p,
            )
            item = items.get(pid)
            if item is not None:
                st.markdown(
                    f"**Category:** {item.category.value} · **Expected:** "
                    f"{item.expected_behavior.value}"
                )
                st.markdown("**Prompt**")
                st.code(item.prompt, language=None, wrap_lines=True)
                if item.context:
                    with st.expander("Context"):
                        st.write(item.context)
                st.markdown(f"**Reference answer:** {item.reference_answer}")
            rows = view[view["prompt_id"] == pid]
            repeats = sorted(rows["repeat_idx"].unique())
            rep = st.radio("Repeat", repeats, horizontal=True, format_func=lambda r: f"#{r + 1}")
            models_here = [m for m in all_models if m in set(rows["model_name"])]
            model_cols = st.columns(len(models_here))
            for col, model in zip(model_cols, models_here, strict=True):
                r = rows[(rows["model_name"] == model) & (rows["repeat_idx"] == rep)]
                with col:
                    st.markdown(
                        f"<span style='color:{COLORS[model]}'>●</span> **{model}**",
                        unsafe_allow_html=True,
                    )
                    if r.empty:
                        st.caption("no result")
                        continue
                    row = r.iloc[0]
                    if pd.notna(row["error_type"]):
                        st.error(f"{row['error_type']}: {row['error_message']}")
                        continue
                    q = row["quality_score"]
                    st.metric("Quality", f"{q:.2f}" if pd.notna(q) else "n/a")
                    st.text_area(
                        "Response",
                        row["response_text"],
                        height=180,
                        key=f"resp-{model}-{pid}-{rep}",
                        disabled=True,
                    )
                    tokens = f"{row['input_tokens']} in / {row['output_tokens']} out"
                    if row["tokens_estimated"]:
                        tokens += " (estimated)"
                    ttft = f" · TTFT {row['ttft_ms']:.0f} ms" if pd.notna(row["ttft_ms"]) else ""
                    cost = fmt_money(row["cost_list_usd"])
                    st.caption(f"{row['latency_ms']:.0f} ms{ttft} · {tokens} · list {cost}")
                    m = metrics[metrics["result_id"] == row["result_id"]]
                    st.dataframe(
                        m[["name", "value", "source", "reason"]],
                        hide_index=True,
                        use_container_width=True,
                    )

# ----------------------------------------------------------------------------- 6 history
with tabs[5]:
    per_run = (
        results[results["model_name"].isin(models_sel)]
        .pipe(run_timeseries)
        .pivot_table(index="run_id", columns="model_name", values="quality")
    )
    hist = runs.merge(per_run, left_on="run_id", right_index=True, how="left")
    show_cols = [
        "run_id",
        "started_at",
        "status",
        "label",
        "git_commit",
        "is_baseline",
        "synthetic",
        *[m for m in all_models if m in hist.columns],
    ]
    st.dataframe(hist[show_cols].iloc[::-1], hide_index=True, use_container_width=True)
    st.caption("Model columns show mean quality per run.")
    pick = st.selectbox("Config snapshot for run", runs["run_id"].tolist()[::-1])
    st.json(json.loads(runs.set_index("run_id").loc[pick, "config_json"]), expanded=False)
