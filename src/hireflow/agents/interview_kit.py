from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hireflow.models.application import Application
from hireflow.models.interview import Interview
from hireflow.models.job import Job
from hireflow.models.screening import ScreeningScore


class KitQuestion(BaseModel):
    topic: str = Field(description="The skill or rubric item this question checks")
    question: str = Field(description="One open question asking for a concrete past example")
    good_answer_signals: str = Field(description="What a strong answer would include")


class InterviewKit(BaseModel):
    focus_areas: list[str] = Field(description="2-4 areas the interviewer should probe most")
    questions: list[KitQuestion] = Field(min_length=3, max_length=8)


KIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You prepare a structured interview for the role: {job_title}.\n"
            "Must-have skills: {must_have}\n"
            "Screening found these WEAK or UNVERIFIED areas: {gaps}\n"
            "Screening found these STRONG areas: {strengths}\n\n"
            "Write 5-6 questions. Start with the weak or unverified areas, because the interview "
            "must confirm them. Ask for concrete examples from past work. Never ask about age, "
            "family, religion, nationality, health or other personal matters.",
        ),
        ("human", "Screening summary of this candidate: {summary}"),
    ]
)


async def create_interview_kit(
    model: BaseChatModel, session_factory: async_sessionmaker[AsyncSession], interview_id: int
) -> dict:
    """A second, small agent: reads the screening results and writes targeted questions."""
    async with session_factory() as session:
        interview = await session.get(Interview, interview_id)
        
        if interview is None:
            raise ValueError(f"Interview {interview_id} not found")
        app = await session.get(Application, interview.application_id)
        
        if app is None:
            raise ValueError(f"Application {interview.application_id} not found")
        job = await session.get(Job, app.job_id)
        
        if job is None:
            raise ValueError(f"Job {app.job_id} not found")
        latest = (
            await session.execute(
                select(ScreeningScore)
                .where(ScreeningScore.application_id == app.id)
                .order_by(ScreeningScore.created_at.desc(), ScreeningScore.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    criteria = latest.details["criteria"] if latest else []
    gaps = [c["criterion"] for c in criteria if c["score"] < 3 or not c["verified"]]
    strengths = [c["criterion"] for c in criteria if c["score"] >= 3 and c["verified"]]
    if not criteria:  
        gaps = list(job.must_have)

    chain = KIT_PROMPT | model.with_structured_output(InterviewKit)
    kit = await chain.ainvoke(
        {
            "job_title": job.title,
            "must_have": ", ".join(job.must_have) or "none listed",
            "gaps": ", ".join(gaps) or "none",
            "strengths": ", ".join(strengths) or "none",
            "summary": latest.details.get("summary", "") if latest else "Not screened.",
        }
    )
    
    if not isinstance(kit, InterviewKit):
        raise TypeError(f"Expected InterviewKit from the LLM, got {type(kit).__name__}")

    async with session_factory() as session:
        interview = await session.get(Interview, interview_id)
        
        if interview is None:
            raise ValueError(f"Interview {interview_id} not found")
        
        kit_data = kit.model_dump()
        interview.kit = kit_data
        await session.commit()
    return kit_data