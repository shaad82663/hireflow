from pydantic import BaseModel, EmailStr, Field, ValidationError, field_validator, model_validator


class Application(BaseModel):
    name: str = Field(min_length=2)
    email: EmailStr
    years: float = Field(ge=0, le=50)
    skills: list[str] = []
    expected_ctc_lpa: float | None = None
    notice_days: int = 0

    @field_validator("skills")
    @classmethod
    def lowercase_skills(cls, v: list[str]) -> list[str]:
        return [s.strip().lower() for s in v if s.strip()]

    def check_notice(self) -> "Application":
        if self.notice_days > 90:
            raise ValueError("notice period above 90 days is not accepted")
        return self


app = Application(name="Asha", email="asha@example.com", years=4, skills=[" Python ", ""])
print(app.skills)

print(app.model_dump())
print(app.model_dump_json())

raw = '{"name": "Ravi", "email": "ravi@example.com", "years": 7}'
ravi = Application.model_validate_json(raw)
print(ravi)

try:
    Application(name="A", email="not-an-email", years=-1)
except ValidationError as e:
    print(e.error_count(), "errors")
    for err in e.errors():
        print(err["loc"], err["msg"])