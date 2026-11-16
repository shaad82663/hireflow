import enum

from sqlalchemy import JSON, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hireflow.db.base import Base, TimestampMixin


class ScreeningDecision(enum.StrEnum):
    SHORTLISTED = "shortlisted"
    REJECTED = "rejected"
    UNDECIDED = "undecided"


class ScreeningScore(TimestampMixin, Base):
    """The audit record: what the AI scored, and what the human decided. Never edited later."""

    __tablename__ = "screening_scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(100), index=True)  # which screening run
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    total: Mapped[float]
    must_haves_met: Mapped[bool]
    details: Mapped[dict] = mapped_column(JSON)  # per-criterion scores, evidence, summary
    decision: Mapped[ScreeningDecision] = mapped_column(Enum(ScreeningDecision, native_enum=False))
    reviewer_note: Mapped[str | None] = mapped_column(Text)