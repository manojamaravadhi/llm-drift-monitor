"""Drift detection: compare a run against a baseline with a practical threshold AND a CI.

A finding is flagged only when the change is both large enough to matter (threshold) and
unlikely to be noise (the bootstrap confidence interval excludes zero). Groups with too few
prompts are reported as ``insufficient_data`` and never flagged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from llmdrift.analysis import per_prompt_scores
from llmdrift.config import DriftConfig
from llmdrift.metrics.stats import (
    bootstrap_paired_mean_diff,
    bootstrap_stat_diff,
    mean,
    p95,
)

Status = Literal["drift", "ok", "insufficient_data"]
ALL = "ALL"


@dataclass
class Finding:
    """Result of one comparison (one model, one category, one metric)."""

    model: str
    category: str
    metric: str
    status: Status
    baseline: float | None
    current: float | None
    delta: float | None
    ci_low: float | None
    ci_high: float | None
    threshold: float
    n: int
    message: str


@dataclass
class DriftReport:
    """Full drift report for one current run."""

    current_run: str
    baseline_mode: str
    baseline_runs: list[str]
    settings: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def drift_detected(self) -> bool:
        """True if any finding is flagged."""
        return any(f.status == "drift" for f in self.findings)

    @property
    def flagged(self) -> list[Finding]:
        """Only the flagged findings."""
        return [f for f in self.findings if f.status == "drift"]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "generated_at": self.generated_at,
            "current_run": self.current_run,
            "baseline": {"mode": self.baseline_mode, "runs": self.baseline_runs},
            "drift_detected": self.drift_detected,
            "n_flagged": len(self.flagged),
            "settings": self.settings,
            "findings": [asdict(f) for f in self.findings],
        }


def _baseline_prompt_scores(baseline: pd.DataFrame) -> pd.DataFrame:
    """Per (model, category, prompt) baseline score = median across baseline runs."""
    ps = per_prompt_scores(baseline)
    return ps.groupby(["model_name", "category", "prompt_id"])["score"].median().reset_index()


def _quality_finding(
    model: str, category: str, cur: pd.DataFrame, base: pd.DataFrame, cfg: DriftConfig
) -> Finding:
    merged = cur.merge(base, on="prompt_id", suffixes=("_cur", "_base"))
    n = len(merged)
    if n < cfg.min_prompts:
        return Finding(
            model, category, "quality", "insufficient_data", None, None, None, None, None,
            cfg.quality_threshold, n, f"only {n} paired prompts (< {cfg.min_prompts})",
        )  # fmt: skip
    res = bootstrap_paired_mean_diff(
        merged["score_cur"].tolist(),
        merged["score_base"].tolist(),
        n_samples=cfg.bootstrap_samples,
        confidence=cfg.confidence,
        seed=cfg.seed,
    )
    drift = res.delta <= -cfg.quality_threshold and res.ci_high < 0
    base_mean = float(merged["score_base"].mean())
    cur_mean = float(merged["score_cur"].mean())
    msg = (
        f"quality {base_mean:.3f} -> {cur_mean:.3f} (delta {res.delta:+.3f}, "
        f"{cfg.confidence:.0%} CI [{res.ci_low:+.3f}, {res.ci_high:+.3f}])"
    )
    return Finding(
        model, category, "quality", "drift" if drift else "ok", base_mean, cur_mean,
        res.delta, res.ci_low, res.ci_high, cfg.quality_threshold, n, msg,
    )  # fmt: skip


def _increase_finding(
    model: str,
    metric: str,
    cur_vals: list[float],
    base_vals: list[float],
    stat_name: Literal["p95", "mean"],
    relative: bool,
    threshold: float,
    cfg: DriftConfig,
    min_n: int,
) -> Finding:
    """Flag an *increase* in a statistic (latency, cost, error rate, refusal rate)."""
    n = min(len(cur_vals), len(base_vals))
    if n < min_n:
        return Finding(
            model, ALL, metric, "insufficient_data", None, None, None, None, None,
            threshold, n, f"only {n} observations",
        )  # fmt: skip
    stat = p95 if stat_name == "p95" else mean
    res = bootstrap_stat_diff(
        cur_vals, base_vals, stat, cfg.bootstrap_samples, cfg.confidence, cfg.seed
    )
    base_v = stat(np.asarray(base_vals, dtype=float))
    cur_v = stat(np.asarray(cur_vals, dtype=float))
    if relative:
        size = (cur_v / base_v - 1) if base_v > 0 else (float("inf") if cur_v > 0 else 0.0)
    else:
        size = cur_v - base_v
    drift = size > threshold and res.ci_low > 0
    unit = "relative" if relative else "absolute"
    msg = (
        f"{metric} {base_v:.4g} -> {cur_v:.4g} ({unit} change {size:+.2%} vs threshold "
        f"{threshold:.0%}; CI of diff [{res.ci_low:+.4g}, {res.ci_high:+.4g}])"
    )
    return Finding(
        model, ALL, metric, "drift" if drift else "ok", base_v, cur_v, res.delta,
        res.ci_low, res.ci_high, threshold, n, msg,
    )  # fmt: skip


def detect_drift(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    cfg: DriftConfig,
    current_run: str,
    baseline_runs: list[str],
    baseline_mode: str,
) -> DriftReport:
    """Compare ``current`` results (one run) to ``baseline`` results (one or more runs)."""
    report = DriftReport(
        current_run=current_run,
        baseline_mode=baseline_mode,
        baseline_runs=baseline_runs,
        settings=cfg.model_dump(),
    )
    cur_ps = per_prompt_scores(current)
    base_ps = _baseline_prompt_scores(baseline)
    models = sorted(set(current["model_name"]) & set(baseline["model_name"]))
    for model in models:
        cur_m = cur_ps[cur_ps["model_name"] == model]
        base_m = base_ps[base_ps["model_name"] == model]
        report.findings.append(
            _quality_finding(model, ALL, cur_m[["prompt_id", "score"]], base_m, cfg)
        )
        for category in sorted(set(cur_m["category"])):
            report.findings.append(
                _quality_finding(
                    model,
                    category,
                    cur_m[cur_m["category"] == category][["prompt_id", "score"]],
                    base_m[base_m["category"] == category][["prompt_id", "score"]],
                    cfg,
                )
            )

        cur_r = current[current["model_name"] == model]
        base_r = baseline[baseline["model_name"] == model]
        cur_ok, base_ok = cur_r[cur_r["error_type"].isna()], base_r[base_r["error_type"].isna()]
        report.findings.append(
            _increase_finding(
                model, "latency_p95_ms",
                cur_ok["latency_ms"].dropna().tolist(), base_ok["latency_ms"].dropna().tolist(),
                "p95", True, cfg.latency_threshold, cfg, cfg.min_prompts,
            )
        )  # fmt: skip
        report.findings.append(
            _increase_finding(
                model, "cost_list_per_query_usd",
                cur_ok["cost_list_usd"].dropna().tolist(),
                base_ok["cost_list_usd"].dropna().tolist(),
                "mean", True, cfg.cost_threshold, cfg, cfg.min_prompts,
            )
        )  # fmt: skip
        report.findings.append(
            _increase_finding(
                model, "error_rate",
                cur_r["error_type"].notna().astype(float).tolist(),
                base_r["error_type"].notna().astype(float).tolist(),
                "mean", False, cfg.rate_threshold, cfg, cfg.min_prompts,
            )
        )  # fmt: skip
        cur_ans = cur_ok[cur_ok["expected_behavior"] == "answer"]
        base_ans = base_ok[base_ok["expected_behavior"] == "answer"]
        report.findings.append(
            _increase_finding(
                model, "over_refusal_rate",
                cur_ans["refused"].astype(float).tolist(),
                base_ans["refused"].astype(float).tolist(),
                "mean", False, cfg.rate_threshold, cfg, cfg.min_prompts,
            )
        )  # fmt: skip
    return report


def to_markdown(report: DriftReport) -> str:
    """Human-readable summary of a drift report."""
    status = "DRIFT DETECTED" if report.drift_detected else "No drift detected"
    lines = [
        f"# Drift report: {status}",
        "",
        f"- Current run: `{report.current_run}`",
        f"- Baseline ({report.baseline_mode}): {', '.join(f'`{r}`' for r in report.baseline_runs)}",
        f"- Generated: {report.generated_at}",
        f"- Rule: flagged only if the change exceeds the threshold **and** the "
        f"{report.settings.get('confidence', 0.95):.0%} bootstrap CI excludes zero.",
        "",
    ]
    if report.flagged:
        lines += ["## Flagged", "", "| Model | Category | Metric | Detail |", "|---|---|---|---|"]
        lines += [
            f"| {f.model} | {f.category} | {f.metric} | {f.message} |" for f in report.flagged
        ]
        lines.append("")
    others = [f for f in report.findings if f.status != "drift"]
    if others:
        lines += ["## Not flagged", "", "| Model | Category | Metric | Status | Detail |"]
        lines.append("|---|---|---|---|---|")
        lines += [
            f"| {f.model} | {f.category} | {f.metric} | {f.status} | {f.message} |" for f in others
        ]
    lines += [
        "",
        "_Small samples (about 8 prompts per category) can only detect large regressions. "
        "Judge-based scores are noisy; see docs/DESIGN.md._",
    ]
    return "\n".join(lines) + "\n"
