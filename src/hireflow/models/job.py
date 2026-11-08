import enum


from sqlalchemy import JSON, Enum, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hireflow.db.base import Base, TimestampMixin

class JobStatus(enum.StrEnum):
    DRAFT = "draft"
    OPEN = "open"
    CLOSED = "closed"

class EmploymentType(enum.StrEnum):
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACTOR = "CONTRACTOR"
    INTERN = "INTERN"


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(220), unique=True, index=True)
    location: Mapped[str] = mapped_column(String(120))
    remote: Mapped[bool] = mapped_column(default=False)
    employment_type: Mapped[EmploymentType] = mapped_column(
        Enum(EmploymentType, native_enum=False), default=EmploymentType.FULL_TIME
    )
    description: Mapped[str] = mapped_column(Text)
    must_have: Mapped[list[str]] = mapped_column(JSON, default=list)
    nice_to_have: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False), default=JobStatus.DRAFT, index=True
    )