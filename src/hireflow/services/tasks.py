from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from hireflow.models.task import Task, TaskStatus

BACKOFF_BASE_SECONDS = 5
BACKOFF_MAX_SECONDS = 600

async def enqueue(
    session: AsyncSession,
    kind: str,
    payload: dict,
    *,
    idempotency_key: str | None = None,
    run_after: datetime | None = None,
) -> int | None:
    """Add a task. Does NOT commit: call it inside the same transaction as the data change
    that needs the work, so you never get 'data saved but task lost' (or the reverse).
    Returns the new task id, or None if a task with this idempotency_key already exists."""
    values = {"kind": kind, "payload": payload, "idempotency_key": idempotency_key}
    if run_after is not None:
        values["run_after"] = run_after
    stmt = (
        insert(Task)
        .values(**values)
        .on_conflict_do_nothing(index_elements=["idempotency_key"])
        .returning(Task.id)
    )
    return (await session.execute(stmt)).scalar_one_or_none()

async def claim_next(session: AsyncSession) -> Task | None:
    """Take the oldest due task. FOR UPDATE SKIP LOCKED makes this safe with many workers:
    a row another worker is claiming right now is skipped instead of waited for."""
    stmt = (
        select(Task)
        .where(Task.status == TaskStatus.QUEUED, Task.run_after <= func.now())
        .order_by(Task.run_after, Task.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    task = (await session.execute(stmt)).scalar_one_or_none()
    if task is None:
        return None
    task.status = TaskStatus.RUNNING
    task.attempts += 1
    await session.commit()
    return task


async def mark_done(session: AsyncSession, task_id: int) -> None:
    task = await session.get(Task, task_id)
    if task is None:
        return None
    task.status = TaskStatus.DONE
    task.last_error = None
    await session.commit()

def backoff(attempts: int) -> timedelta:
    """5s, 10s, 20s, 40s ... capped at 10 minutes: retry soon, but don't hammer a broken service."""
    return timedelta(seconds=min(BACKOFF_BASE_SECONDS * 2 ** (attempts - 1), BACKOFF_MAX_SECONDS))

async def mark_failed(session: AsyncSession, task_id: int, error: str) -> TaskStatus:
    """Retry later with backoff, or give up after max_attempts."""
    task = await session.get(Task, task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")
    task.last_error = error[:2000]
    if task.attempts >= task.max_attempts:
        task.status = TaskStatus.FAILED  # a human looks at FAILED tasks
    else:
        task.status = TaskStatus.QUEUED
        task.run_after = datetime.now(UTC) + backoff(task.attempts)
    await session.commit()
    return task.status