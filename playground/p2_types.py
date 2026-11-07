from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol



def avg(scores: list[float]) -> float | None:
    return sum(scores) / len(scores) if scores else None


class Recruiter:
    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    def signature(self):
        return f"{self.name} - {self.email}"

    def __repr__(self) -> str:
        return f"Recruiter({self.name!r})"

@dataclass
class Candidate:
    name: str
    years: float
    skills: list[str] = field(default_factory=list)

    def has(self, skill: str):
        return skill.lower() in (s.lower() for s in self.skills)


class Stage(StrEnum):
    APPLIED = "applied"
    SHORTLISTED = "shortlisted"
    INTERVIEW = "interview"
    REJECTED = "rejected"

class Notifier(Protocol):
    def send(self, to: str, message: str) -> None: ...