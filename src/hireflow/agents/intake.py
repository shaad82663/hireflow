from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

from hireflow.models.job import EmploymentType


class JobDraft(BaseModel):
    """The structured job post the LLM must return. Field descriptions are sent to the LLM."""

    title: str = Field(description="Clear, standard job title, e.g. 'Senior Backend Engineer'")
    location: str = Field(description="City or region. Use 'Remote' if fully remote.")
    remote: bool = Field(description="True if the role can be done remotely")
    employment_type: EmploymentType
    description: str = Field(description="Friendly 120-200 word job description")
    must_have: list[str] = Field(description="3-6 hard requirements, short phrases")
    nice_to_have: list[str] = Field(description="0-5 optional skills, short phrases")
    biased_phrases_found: list[str] = Field(
        default_factory=list,
        description="Phrases from the notes that could discourage some groups, e.g. 'young team'",
    )


INTAKE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert recruiter at {company}. Turn a hiring manager's rough notes into a "
            "clear, inclusive job post. Never invent salary numbers or benefits that are not in "
            "the notes. Do not copy biased wording into the post; list it in biased_phrases_found.",
        ),
        ("human", "Hiring manager notes:\n\n{notes}"),
    ]
)

async def draft_job(model: BaseChatModel, notes: str, company: str) -> JobDraft:
    chain = INTAKE_PROMPT | model.with_structured_output(JobDraft)
    result = await chain.ainvoke({"notes": notes, "company": company})
    if not isinstance(result, JobDraft):  # narrows the type for Pylance, and guards at runtime
        raise TypeError(f"Expected JobDraft from the LLM, got {type(result).__name__}")
    return result