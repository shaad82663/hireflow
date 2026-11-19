from langchain_core.tools import BaseTool, tool
from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hireflow.services import interviews as interview_service
from hireflow.services.jobs import JobNotFoundError

_email = TypeAdapter(EmailStr)

def build_interview_tools(session_factory: async_sessionmaker[AsyncSession]) -> list[BaseTool]:
    """Tools that let the recruiting assistant (the 'supervisor') hand work to the
    scheduling agent. The scheduling agent itself runs in the worker, not inside this chat."""

    @tool
    async def schedule_interviews(job_id: int, interviewer_email: str) -> dict:
        """Start scheduling interviews for every SHORTLISTED candidate of a job. Invitation
        emails are sent in the background. Only use when the recruiter asks for it."""
        try:
            _email.validate_python(interviewer_email)
        except ValidationError:
            return {"error": f"'{interviewer_email}' is not a valid email address"}
        async with session_factory() as session:
            try:
                created = await interview_service.start_scheduling_for_job(
                    session, job_id, interviewer_email
                )
            except JobNotFoundError as e:
                return {"error": str(e)}
        return {"started": len(created)}

    @tool
    async def interview_status(job_id: int) -> list[dict]:
        """Show the interview scheduling status of every candidate for a job: scheduling,
        booked (with the time), needs_human, or withdrawn."""
        async with session_factory() as session:
            rows = await interview_service.list_interviews(session, job_id)
        return rows or [{"info": "No interviews for this job yet."}]

    

    return [schedule_interviews, interview_status]