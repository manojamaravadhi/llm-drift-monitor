"""SQLite schema and data-access helpers (SQLAlchemy 2.x ORM)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """ORM base class."""


class Run(Base):
    """One evaluation run (all models, all prompts, N repeats)."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="running")
    git_commit: Mapped[str | None] = mapped_column(String(64))
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    config_hash: Mapped[str] = mapped_column(String(64), index=True)
    dataset_hash: Mapped[str] = mapped_column(String(64))
    label: Mapped[str | None] = mapped_column(String(200))
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)

    results: Mapped[list[Result]] = relationship(back_populates="run", cascade="all, delete")

    @property
    def config(self) -> dict[str, Any]:
        """Decoded config snapshot."""
        data = json.loads(self.config_json)
        return data if isinstance(data, dict) else {}


class Result(Base):
    """One model response to one prompt (one repeat)."""

    __tablename__ = "results"
    __table_args__ = (UniqueConstraint("run_id", "model_name", "prompt_id", "repeat_idx"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    model_id: Mapped[str] = mapped_column(String(200))
    prompt_id: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(32))
    expected_behavior: Mapped[str] = mapped_column(String(16), default="answer")
    repeat_idx: Mapped[int] = mapped_column(Integer)
    response_text: Mapped[str | None] = mapped_column(Text)
    error_type: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    ttft_ms: Mapped[float | None] = mapped_column(Float)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    tokens_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_billed_usd: Mapped[float | None] = mapped_column(Float)
    cost_list_usd: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float | None] = mapped_column(Float)
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    from_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_key: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[Run] = relationship(back_populates="results")
    metrics: Mapped[list[MetricRow]] = relationship(
        back_populates="result", cascade="all, delete-orphan"
    )


class MetricRow(Base):
    """One named metric value for a result."""

    __tablename__ = "metrics"
    __table_args__ = (UniqueConstraint("result_id", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("results.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)

    result: Mapped[Result] = relationship(back_populates="metrics")


class CachedResponse(Base):
    """Provider response cache keyed by hash(model, messages, params, repeat)."""

    __tablename__ = "response_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(200))
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def get_engine(db_path: str | Path, create: bool = True) -> Engine:
    """Engine for a SQLite file (``:memory:`` supported). Creates tables when ``create``."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    event.listen(engine, "connect", _sqlite_pragmas)
    if create:
        Base.metadata.create_all(engine)
    return engine


def latest_resumable_run(session: Session, config_hash: str) -> Run | None:
    """Most recent unfinished run with the same config (for resume)."""
    stmt = (
        select(Run)
        .where(Run.status == "running", Run.config_hash == config_hash)
        .order_by(Run.started_at.desc())
    )
    return session.scalars(stmt).first()


def completed_runs(session: Session) -> list[Run]:
    """All completed runs, oldest first."""
    stmt = select(Run).where(Run.status == "completed").order_by(Run.started_at)
    return list(session.scalars(stmt))


def get_cached(session: Session, cache_key: str) -> dict[str, Any] | None:
    """Cached provider response payload, if any."""
    row = session.get(CachedResponse, cache_key)
    if row is None:
        return None
    data = json.loads(row.payload_json)
    return data if isinstance(data, dict) else None


def put_cached(session: Session, cache_key: str, model_id: str, payload: dict[str, Any]) -> None:
    """Store (or replace) a cached provider response."""
    session.merge(
        CachedResponse(cache_key=cache_key, model_id=model_id, payload_json=json.dumps(payload))
    )


def set_baseline(session: Session, run_id: str) -> None:
    """Mark exactly one run as the designated baseline."""
    run = session.get(Run, run_id)
    if run is None:
        raise KeyError(run_id)
    for other in session.scalars(select(Run).where(Run.is_baseline.is_(True))):
        other.is_baseline = False
    run.is_baseline = True
