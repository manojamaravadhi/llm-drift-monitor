"""Pick the current run and its baseline from the database."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from llmdrift.config import DriftConfig
from llmdrift.storage.db import Run, completed_runs


class BaselineError(ValueError):
    """No usable baseline (e.g. only one run so far)."""


@dataclass(frozen=True)
class BaselineChoice:
    """The run under test and the run(s) it is compared with."""

    current_run: str
    baseline_runs: list[str]
    mode: str  # "explicit", "designated" or "rolling-K"


def choose_baseline(
    engine: Engine,
    cfg: DriftConfig,
    current_run: str | None = None,
    baseline_run: str | None = None,
) -> BaselineChoice:
    """Resolve current and baseline runs.

    Priority: ``baseline_run`` argument, then ``cfg.baseline`` if it names a run, then a run
    marked ``is_baseline``, then the ``rolling_k`` completed runs before the current one.
    """
    with Session(engine) as session:
        runs = completed_runs(session)
        if not runs:
            raise BaselineError("no completed runs in the database")
        ids = [r.run_id for r in runs]
        current = current_run or ids[-1]
        if current not in ids:
            raise BaselineError(f"run {current!r} not found or not completed")
        earlier = ids[: ids.index(current)]

        explicit = baseline_run or (cfg.baseline if cfg.baseline != "rolling" else None)
        if explicit:
            if session.get(Run, explicit) is None:
                raise BaselineError(f"baseline run {explicit!r} not found")
            if explicit == current:
                raise BaselineError("baseline and current run are the same")
            return BaselineChoice(current, [explicit], "explicit")

        designated = [r.run_id for r in runs if r.is_baseline and r.run_id != current]
        if designated:
            return BaselineChoice(current, designated[-1:], "designated")

        if not earlier:
            raise BaselineError("need at least one earlier completed run for a rolling baseline")
        window = earlier[-cfg.rolling_k :]
        return BaselineChoice(current, window, f"rolling-{len(window)}")
