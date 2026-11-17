import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from hireflow.db.base import Base, TimestampMixin

class TaskStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class Task(TimestampMixin, Base):
    """One unit of background work, stored in Postgres so it survives restarts."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(50), index=True)  # which handler runs it
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, native_enum=False), default=TaskStatus.QUEUED, index=True
    )
    attempts: Mapped[int] = mapped_column(default=0)
    max_attempts: Mapped[int] = mapped_column(default=5)
    run_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    # Same key twice = same task. Stops duplicates from double clicks and webhook retries.
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True)