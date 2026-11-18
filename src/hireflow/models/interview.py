import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from hireflow.db.base import Base, TimestampMixin

class InterviewStatus(enum.StrEnum):
    SCHEDULING = "scheduling"  # the scheduling agent is talking to the candidate
    BOOKED = "booked"
    NEEDS_HUMAN = "needs_human"  # the agent gave up and asked a recruiter to step in
    WITHDRAWN = "withdrawn"  # the candidate declined


class Interview(TimestampMixin, Base):
    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), unique=True
    )
    interviewer_email: Mapped[str] = mapped_column(String(254))
    thread_id: Mapped[str] = mapped_column(String(100), unique=True)  # the LangGraph run
    reply_token: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    status: Mapped[InterviewStatus] = mapped_column(
        Enum(InterviewStatus, native_enum=False), default=InterviewStatus.SCHEDULING
    )
    starts_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    kit: Mapped[dict | None] = mapped_column(JSON)  # questions from the interview-kit agent

class CalendarBlock(Base):
    """A busy period in someone's calendar."""

    __tablename__ = "calendar_blocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(254), index=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(String(200))