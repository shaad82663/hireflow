from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hireflow.models.interview import CalendarBlock
from hireflow.scheduling.slots import Interval


class Calendar(Protocol):
    """Anything that can tell when someone is busy and book time. Our DbCalendar implements it;
    a Google or Microsoft calendar adapter would implement the same two methods."""

    async def busy(self, email: str, start: datetime, end: datetime) -> list[Interval]: ...

    async def book(self, email: str, start: datetime, end: datetime, reason: str) -> None: ...


class DbCalendar:
    """A simple calendar stored in our own database (the calendar_blocks table)."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def busy(self, email: str, start: datetime, end: datetime) -> list[Interval]:
        stmt = select(CalendarBlock.starts_at, CalendarBlock.ends_at).where(
            CalendarBlock.email == email,
            CalendarBlock.starts_at < end,  # overlap test, same rule as slots.overlaps
            CalendarBlock.ends_at > start,
        )
        async with self.session_factory() as session:
            return [(a, b) for a, b in (await session.execute(stmt)).all()]

    async def book(self, email: str, start: datetime, end: datetime, reason: str) -> None:
        async with self.session_factory() as session:
            session.add(CalendarBlock(email=email, starts_at=start, ends_at=end, reason=reason))
            await session.commit()