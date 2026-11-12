import enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hireflow.core.config import get_settings
from hireflow.db.base import Base, TimestampMixin


class ApplicationStage(enum.StrEnum):
    APPLIED = "applied"
    SHORTLISTED = "shortlisted"
    INTERVIEW = "interview"
    REJECTED = "rejected"
    HIRED = "hired"


class Candidate(TimestampMixin, Base):
    """A person. One candidate can apply to many jobs."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)

    applications: Mapped[list["Application"]] = relationship(back_populates="candidate")


class Application(TimestampMixin, Base):
    """One candidate applying to one job, with the resume they sent."""

    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint("job_id", "candidate_id", name="uq_application_job_candidate"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"))
    resume_filename: Mapped[str] = mapped_column(String(255))
    resume_text: Mapped[str] = mapped_column(Text)
    consent_given: Mapped[bool]
    stage: Mapped[ApplicationStage] = mapped_column(
        Enum(ApplicationStage, native_enum=False), default=ApplicationStage.APPLIED, index=True
    )

    candidate: Mapped[Candidate] = relationship(back_populates="applications")
    chunks: Mapped[list["ResumeChunk"]] = relationship(
        back_populates="application", cascade="all, delete-orphan"
    )

class ResumeChunk(Base):
    """A small piece of a resume plus its embedding vector, for search."""

    __tablename__ = "resume_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(get_settings().embedding_dim))

    application: Mapped[Application] = relationship(back_populates="chunks")