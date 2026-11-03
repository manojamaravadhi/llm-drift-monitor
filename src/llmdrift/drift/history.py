"""Drift status for every run in history (used by the dashboard time series)."""

from __future__ import annotations

import pandas as pd

from llmdrift.analysis import per_prompt_scores
from llmdrift.config import DriftConfig
from llmdrift.drift.detector import ALL, detect_drift


def drift_history(results: pd.DataFrame, runs: pd.DataFrame, cfg: DriftConfig) -> pd.DataFrame:
    """For each completed run, compare it with the ``rolling_k`` runs before it.

    Returns one row per (run, model) with the baseline mean quality, the band
    ``baseline ± quality_threshold`` and which metrics were flagged.
    """
    columns = ["run_id", "model_name", "baseline_quality", "band_low", "band_high", "flagged"]
    done = runs[runs["status"] == "completed"].sort_values("started_at")
    ids = done["run_id"].tolist()
    rows = []
    for i, run_id in enumerate(ids[1:], start=1):
        window = ids[max(0, i - cfg.rolling_k) : i]
        current = results[results["run_id"] == run_id]
        baseline = results[results["run_id"].isin(window)]
        if current.empty or baseline.empty:
            continue
        report = detect_drift(current, baseline, cfg, run_id, window, f"rolling-{len(window)}")
        base_scores = per_prompt_scores(baseline)
        for model in sorted(set(current["model_name"])):
            per_prompt = base_scores[base_scores["model_name"] == model]
            if per_prompt.empty:
                continue
            by_prompt = per_prompt.groupby("prompt_id")["score"]
            agg = by_prompt.median() if cfg.rolling_stat == "median" else by_prompt.mean()
            base_q = float(agg.mean())
            flagged = sorted(
                f.metric if f.category == ALL else f"{f.metric}:{f.category}"
                for f in report.flagged
                if f.model == model
            )
            rows.append(
                {
                    "run_id": run_id,
                    "model_name": model,
                    "baseline_quality": base_q,
                    "band_low": base_q - cfg.quality_threshold,
                    "band_high": base_q + cfg.quality_threshold,
                    "flagged": ", ".join(flagged),
                }
            )
    return pd.DataFrame(rows, columns=columns)
