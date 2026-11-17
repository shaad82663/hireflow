import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from hireflow.db.session import SessionLocal
from hireflow.models.task import Task, TaskStatus
from hireflow.services.tasks import backoff, claim_next, enqueue, mark_failed
from hireflow.worker import Worker


async def add(kind="demo", payload=None, **kwargs) -> int | None:
    async with SessionLocal() as session:
        task_id = await enqueue(session, kind, payload or {}, **kwargs)
        await session.commit()
        return task_id


async def get(task_id: int) -> Task:
    async with SessionLocal() as session:
        return await session.get(Task, task_id)


async def test_idempotency_key_prevents_duplicates():
    first = await add(idempotency_key="send-invite-7")
    second = await add(idempotency_key="send-invite-7")
    assert first is not None and second is None
    async with SessionLocal() as session:
        assert len((await session.execute(select(Task))).scalars().all()) == 1


async def test_claim_takes_oldest_due_task_and_skips_future_ones():
    await add(payload={"n": 1}, run_after=datetime.now(UTC) + timedelta(hours=1))
    due = await add(payload={"n": 2})
    async with SessionLocal() as session:
        task = await claim_next(session)
    assert task.id == due and task.status == TaskStatus.RUNNING and task.attempts == 1
    async with SessionLocal() as session:
        assert await claim_next(session) is None  # the other one is not due yet


async def test_two_workers_never_claim_the_same_task():
    await add(payload={"n": 1})
    await add(payload={"n": 2})

    async def claim_in_own_session():
        async with SessionLocal() as session:
            return await claim_next(session)

    a, b = await asyncio.gather(claim_in_own_session(), claim_in_own_session())
    assert a is not None and b is not None and a.id != b.id  # SKIP LOCKED at work


async def test_failures_retry_with_backoff_then_give_up():
    task_id = await add()
    async with SessionLocal() as session:
        task = await session.get(Task, task_id)
        task.max_attempts = 2
        await session.commit()

    async with SessionLocal() as session:
        await claim_next(session)
    async with SessionLocal() as session:
        assert await mark_failed(session, task_id, "boom") == TaskStatus.QUEUED
    task = await get(task_id)
    assert task.run_after > datetime.now(UTC)  # retry later, not immediately

    async with SessionLocal() as session:  # pretend the delay passed
        (await session.get(Task, task_id)).run_after = datetime.now(UTC)
        await session.commit()
    async with SessionLocal() as session:
        await claim_next(session)
    async with SessionLocal() as session:
        assert await mark_failed(session, task_id, "boom again") == TaskStatus.FAILED
    assert (await get(task_id)).last_error == "boom again"


def test_backoff_grows_and_is_capped():
    assert [backoff(n).seconds for n in (1, 2, 3)] == [5, 10, 20]
    assert backoff(30) == timedelta(minutes=10)


async def test_worker_runs_handlers_and_records_failures():
    seen = []

    async def ok(payload):
        seen.append(payload["n"])

    async def broken(payload):
        raise RuntimeError("SMTP server down")

    good = await add("ok", {"n": 1})
    bad = await add("broken", {})
    unknown = await add("no_such_kind", {})

    processed = await Worker(SessionLocal, {"ok": ok, "broken": broken}).run_until_idle()

    assert processed == 3 and seen == [1]
    assert (await get(good)).status == TaskStatus.DONE
    assert (await get(bad)).last_error == "RuntimeError: SMTP server down"
    assert (await get(bad)).status == TaskStatus.QUEUED  # will be retried later
    assert "KeyError" in (await get(unknown)).last_error