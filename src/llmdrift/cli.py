"""Command-line interface: ``llmdrift run | report | drift | export | validate-dataset``."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from llmdrift.analysis import category_matrix, leaderboard, load_metrics, load_results, load_runs
from llmdrift.config import EvalConfig, load_eval_config, load_models, load_pricing
from llmdrift.datasets.schema import load_dataset
from llmdrift.drift.baseline import BaselineError, choose_baseline
from llmdrift.drift.detector import detect_drift, to_markdown
from llmdrift.metrics.judge import DeepEvalJudge, Judge, NullJudge, judge_available
from llmdrift.runner import NoModelsAvailable, Runner, RunOptions
from llmdrift.storage.db import get_engine, set_baseline

app = typer.Typer(
    help="Compare LLMs on a golden dataset and detect quality, latency and cost drift.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # never print locals (could contain keys)
)


@dataclass
class Ctx:
    """Paths shared by all commands."""

    models_path: Path
    eval_path: Path
    pricing_path: Path
    db_override: str | None

    @property
    def eval_cfg(self) -> EvalConfig:
        """Loaded eval.yaml."""
        return load_eval_config(self.eval_path)

    @property
    def db_path(self) -> str:
        """Database path (``--db`` wins over eval.yaml)."""
        return self.db_override or self.eval_cfg.db_path


def _split(values: list[str] | None) -> list[str] | None:
    """Accept both ``--models a,b`` and ``--models a --models b``."""
    if not values:
        return None
    return [v.strip() for value in values for v in value.split(",") if v.strip()]


@app.callback()
def main(
    ctx: typer.Context,
    models_config: Annotated[Path, typer.Option(help="Models YAML.")] = Path("config/models.yaml"),
    eval_config: Annotated[Path, typer.Option(help="Eval settings YAML.")] = Path(
        "config/eval.yaml"
    ),
    pricing_config: Annotated[Path, typer.Option(help="Pricing YAML.")] = Path(
        "config/pricing.yaml"
    ),
    db: Annotated[str | None, typer.Option(help="SQLite path (overrides eval.yaml).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Load .env and shared options."""
    if os.environ.get("LLMDRIFT_DOTENV", "1") != "0":  # tests set 0: never load real keys
        load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # LiteLLM logs request details at INFO; keep it quiet so nothing sensitive is printed.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ctx.obj = Ctx(models_config, eval_config, pricing_config, db)


@app.command("validate-dataset")
def validate_dataset(
    dataset: Annotated[Path, typer.Option(help="JSONL file.")] = Path("datasets/golden.jsonl"),
) -> None:
    """Validate the golden dataset against the schema and print category counts."""
    try:
        items = load_dataset(dataset)
    except (ValueError, OSError) as exc:
        typer.secho(f"Invalid dataset: {exc}", fg="red", err=True)
        raise typer.Exit(1) from exc
    counts = pd.Series([i.category.value for i in items]).value_counts().sort_index()
    behaviours = pd.Series([i.expected_behavior.value for i in items]).value_counts()
    typer.echo(f"{dataset}: {len(items)} valid items")
    typer.echo(counts.to_string())
    typer.echo(behaviours.to_string())


@app.command()
def run(
    ctx: typer.Context,
    models: Annotated[list[str] | None, typer.Option(help="Model names (comma list).")] = None,
    categories: Annotated[list[str] | None, typer.Option(help="Categories (comma list).")] = None,
    runs: Annotated[int | None, typer.Option(help="Repeats per prompt.", min=1)] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass response cache.")] = False,
    fresh: Annotated[bool, typer.Option(help="Start a new run instead of resuming.")] = False,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="Skip LLM-as-judge.")] = False,
    inject_degradation: Annotated[
        bool, typer.Option(help="Use a harmful system prompt to demo drift.")
    ] = False,
    label: Annotated[str | None, typer.Option(help="Free-text label for this run.")] = None,
    mark_baseline: Annotated[bool, typer.Option(help="Mark this run as the baseline.")] = False,
    sequential: Annotated[
        bool, typer.Option("--sequential", help="One model at a time (default: parallel).")
    ] = False,
) -> None:
    """Evaluate models on the golden dataset and store results (resumes if interrupted)."""
    c: Ctx = ctx.obj
    cfg = c.eval_cfg
    dataset = load_dataset(cfg.dataset)
    judge: Judge = NullJudge()
    if not no_judge:
        ok, reason = judge_available(cfg.judge)
        if ok:
            judge = DeepEvalJudge(cfg.judge)
        else:
            typer.secho(f"Judge off: {reason}", fg="yellow")
    engine = get_engine(c.db_path)
    runner = Runner(
        cfg, load_models(c.models_path), load_pricing(c.pricing_path), dataset, engine, judge
    )
    opts = RunOptions(
        models=_split(models),
        categories=_split(categories),
        repeats=runs,
        use_cache=not no_cache,
        fresh=fresh,
        inject_degradation=inject_degradation,
        label=label,
        parallel=not sequential,
    )
    try:
        summary = runner.run(opts)
    except NoModelsAvailable as exc:
        typer.secho(f"{exc}. Set API keys in .env or enable a model.", fg="red", err=True)
        raise typer.Exit(2) from exc
    if mark_baseline:
        with Session(engine) as session:
            set_baseline(session, summary.run_id)
            session.commit()
    verb = "Resumed" if summary.resumed else "Completed"
    typer.secho(f"{verb} run {summary.run_id}", fg="green")
    typer.echo(
        f"models: {', '.join(summary.models_run)} | API calls: {summary.calls_made} | "
        f"cache hits: {summary.cache_hits} | already done: {summary.skipped_existing} | "
        f"errors: {summary.errors}"
    )
    for name, reason in summary.models_skipped.items():
        typer.secho(f"skipped {name}: {reason}", fg="yellow")


def _fmt_leaderboard(df: pd.DataFrame) -> str:
    cols = {
        "model": "Model",
        "quality": "Quality",
        "quality_std_across_repeats": "Std (repeats)",
        "latency_p50_ms": "p50 ms",
        "latency_p95_ms": "p95 ms",
        "cost_billed_per_query_usd": "Billed $/query",
        "cost_list_per_query_usd": "List-price $/query",
        "error_rate": "Error rate",
        "over_refusal_rate": "Over-refusal",
        "hallucination_rate": "Hallucination",
    }
    out = df[list(cols)].rename(columns=cols)
    return out.to_markdown(index=False, floatfmt=".4g")


@app.command()
def report(
    ctx: typer.Context,
    run_id: Annotated[str | None, typer.Option("--run", help="Run id (default: latest).")] = None,
    out: Annotated[Path | None, typer.Option(help="Write markdown here.")] = None,
) -> None:
    """Print a leaderboard and category table for one run."""
    c: Ctx = ctx.obj
    engine = get_engine(c.db_path)
    runs_df = load_runs(engine)
    done = runs_df[runs_df["status"] == "completed"]
    if done.empty:
        typer.secho("No completed runs. Run `llmdrift run` first.", fg="yellow")
        raise typer.Exit(1)
    rid = run_id or str(done.iloc[-1]["run_id"])
    results = load_results(engine, [rid])
    board = leaderboard(results, load_metrics(engine, [rid]))
    matrix = category_matrix(results)
    synthetic = bool(runs_df.set_index("run_id").loc[rid, "synthetic"])
    text = "\n".join(
        [
            f"# Results for run `{rid}`" + (" (SYNTHETIC DEMO DATA)" if synthetic else ""),
            "",
            _fmt_leaderboard(board),
            "",
            "## Quality by category",
            "",
            matrix.round(3).to_markdown(),
            "",
        ]
    )
    typer.echo(text)
    target = out or Path(c.eval_cfg.reports_dir) / "leaderboard.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    typer.secho(f"wrote {target}", fg="green")


@app.command()
def drift(
    ctx: typer.Context,
    run_id: Annotated[
        str | None, typer.Option("--run", help="Run to test (default latest).")
    ] = None,
    baseline: Annotated[str | None, typer.Option(help="Baseline run id.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Report directory.")] = None,
) -> None:
    """Compare a run with its baseline. Exit code 1 if drift is detected."""
    c: Ctx = ctx.obj
    cfg = c.eval_cfg
    engine = get_engine(c.db_path)
    try:
        choice = choose_baseline(engine, cfg.drift, run_id, baseline)
    except BaselineError as exc:
        typer.secho(f"Cannot check drift: {exc}", fg="red", err=True)
        raise typer.Exit(2) from exc
    current = load_results(engine, [choice.current_run])
    base = load_results(engine, choice.baseline_runs)
    rep = detect_drift(
        current, base, cfg.drift, choice.current_run, choice.baseline_runs, choice.mode
    )
    target = out_dir or Path(cfg.reports_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "drift_report.json").write_text(json.dumps(rep.to_dict(), indent=2), "utf-8")
    md = to_markdown(rep)
    (target / "drift_report.md").write_text(md, "utf-8")
    typer.echo(md)
    if rep.drift_detected:
        typer.secho(f"DRIFT: {len(rep.flagged)} finding(s) flagged", fg="red", err=True)
        raise typer.Exit(1)
    typer.secho("No drift detected", fg="green")


@app.command()
def export(
    ctx: typer.Context,
    run_id: Annotated[str | None, typer.Option("--run", help="Run id (default: all).")] = None,
    fmt: Annotated[str, typer.Option("--format", help="csv, jsonl or json.")] = "csv",
    out: Annotated[Path, typer.Option(help="Output file.")] = Path("reports/results.csv"),
) -> None:
    """Export result rows (with metric columns) to CSV / JSONL / JSON."""
    c: Ctx = ctx.obj
    engine = get_engine(c.db_path)
    ids = [run_id] if run_id else None
    results = load_results(engine, ids)
    metrics = load_metrics(engine, ids)
    if not metrics.empty:
        wide = metrics.pivot_table(
            index="result_id", columns="name", values="value", aggfunc="first"
        )
        wide.columns = [f"metric_{col}" for col in wide.columns]
        results = results.merge(wide, left_on="result_id", right_index=True, how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        results.to_csv(out, index=False)
    elif fmt == "jsonl":
        results.to_json(out, orient="records", lines=True, date_format="iso")
    elif fmt == "json":
        results.to_json(out, orient="records", indent=2, date_format="iso")
    else:
        typer.secho(f"unknown format {fmt!r}", fg="red", err=True)
        raise typer.Exit(2)
    typer.secho(f"wrote {len(results)} rows to {out}", fg="green")


@app.command("list-runs")
def list_runs(ctx: typer.Context) -> None:
    """List runs in the database."""
    c: Ctx = ctx.obj
    df = load_runs(get_engine(c.db_path))
    if df.empty:
        typer.echo("No runs yet.")
        return
    cols = ["run_id", "started_at", "status", "git_commit", "label", "is_baseline", "synthetic"]
    typer.echo(df[cols].to_markdown(index=False))


@app.command("set-baseline")
def set_baseline_cmd(ctx: typer.Context, run_id: str) -> None:
    """Designate a run as the drift baseline."""
    c: Ctx = ctx.obj
    with Session(get_engine(c.db_path)) as session:
        try:
            set_baseline(session, run_id)
        except KeyError as exc:
            typer.secho(f"run {run_id} not found", fg="red", err=True)
            raise typer.Exit(2) from exc
        session.commit()
    typer.secho(f"{run_id} is now the baseline", fg="green")


if __name__ == "__main__":  # pragma: no cover
    app()
