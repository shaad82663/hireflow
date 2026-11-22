"""Golden datasets: inputs plus the answer a careful human recruiter would give."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Label = Literal["strong", "maybe", "weak"]
DATA_DIR = Path(__file__).parent / "data"


class EvalJob(BaseModel):
    title: str
    must_have: list[str]
    nice_to_have: list[str] = []


class ScreeningCase(BaseModel):
    id: str
    job: str
    label: Label  # the expected answer, decided by a human
    why: str  # the reason, so the next person can check (and argue with) the label
    name: str
    resume: str = Field(min_length=50)


class ScreeningDataset(BaseModel):
    jobs: dict[str, EvalJob]
    cases: list[ScreeningCase]

    @model_validator(mode="after")
    def every_case_has_a_job(self) -> "ScreeningDataset":
        for case in self.cases:
            if case.job not in self.jobs:
                raise ValueError(f"case {case.id!r} uses unknown job {case.job!r}")
        return self


def load_screening_dataset(path: Path = DATA_DIR / "screening_golden.json") -> ScreeningDataset:
    return ScreeningDataset.model_validate(json.loads(path.read_text()))