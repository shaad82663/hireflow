from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hireflow.models.job import EmploymentType, JobStatus


class JobBase(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    location: str = Field(min_length=2, max_length=120)
    remote: bool = False
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    description: str = Field(min_length=20)
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)

    @field_validator("must_have", "nice_to_have")
    @classmethod
    def clean_skills(cls, skills: list[str]) -> list[str]:
        """Trim spaces and drop empty or duplicate skills, keeping the original order."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for skill in skills:
            s = skill.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                cleaned.append(s)
        return cleaned


class JobCreate(JobBase):
    """What the client sends to create a job."""


class JobUpdate(BaseModel):
    """Every field optional: send only what you want to change."""

    title: str | None = Field(default=None, min_length=3, max_length=200)
    location: str | None = None
    remote: bool | None = None
    employment_type: EmploymentType | None = None
    description: str | None = Field(default=None, min_length=20)
    must_have: list[str] | None = None
    nice_to_have: list[str] | None = None
    status: JobStatus | None = None


class JobRead(JobBase):
    """What the API sends back."""

    model_config = ConfigDict(from_attributes=True)  # lets Pydantic read SQLAlchemy objects

    id: int
    slug: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime