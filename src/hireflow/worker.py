"""The background worker. Run it in its own terminal: uv run python -m hireflow.worker"""

import asyncio
import signal
from collections.abc import Awaitable, Callable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hireflow.services.tasks import claim_next, mark_done, mark_failed

Handler = Callable[[dict], Awaitable[None]]
log = structlog.get_logger()


class Worker:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], handlers: dict[str, Handler]
    ) -> None:
        self.session_factory = session_factory
        self.handlers = handlers

    async def run_once(self) -> bool:
        """Process at most one task. Returns False when there was nothing to do."""
        async with self.session_factory() as session:
            task = await claim_next(session)
        if task is None:
            return False

        bound = log.bind(task_id=task.id, kind=task.kind, attempt=task.attempts)
        try:
            handler = self.handlers[task.kind]
            await handler(task.payload)
        except Exception as e:  # any failure: record it and let the queue retry
            async with self.session_factory() as session:
                status = await mark_failed(session, task.id, f"{type(e).__name__}: {e}")
            bound.warning("task_failed", error=str(e), now=status.value)
        else:
            async with self.session_factory() as session:
                await mark_done(session, task.id)
            bound.info("task_done")
        return True

    async def run_until_idle(self, max_tasks: int = 1000) -> int:
        """Process tasks until none are due. Tests use this instead of a forever loop."""
        count = 0
        while count < max_tasks and await self.run_once():
            count += 1
        return count

    async def run_forever(self, stop: asyncio.Event, poll_seconds: float = 1.0) -> None:
        while not stop.is_set():
            if not await self.run_once():  # queue empty: wait a bit (or until asked to stop)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
                except TimeoutError:
                    pass


async def main() -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from hireflow.core.config import get_settings
    from hireflow.core.logging import setup_logging
    from hireflow.db.session import SessionLocal
    from hireflow.integrations.email import get_email_sender
    from hireflow.llm.client import get_chat_model
    from hireflow.scheduling.handlers import build_handlers

    settings = get_settings()
    setup_logging(settings.debug)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):  # Ctrl+C: finish the current task, then exit
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows: Ctrl+C still raises KeyboardInterrupt
            pass

    async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_db_url) as checkpointer:
        await checkpointer.setup()
        handlers = build_handlers(
            model=get_chat_model(),
            session_factory=SessionLocal,
            email=get_email_sender(),
            checkpointer=checkpointer,
        )
        log.info("worker_started", kinds=sorted(handlers))
        await Worker(SessionLocal, handlers).run_forever(stop)
    log.info("worker_stopped")


if __name__ == "__main__":
    asyncio.run(main())