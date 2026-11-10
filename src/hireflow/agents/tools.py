from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hireflow.models.job import JobStatus
from hireflow.services import jobs as job_service
from hireflow.services.jobs import JobNotDraftError, JobNotFoundError, JobNotOpenError


def build_job_tools(session_factory: async_sessionmaker[AsyncSession]) -> list[BaseTool]:

    @tool
    async def list_jobs(status: JobStatus = JobStatus.OPEN) -> list[dict]:
        """List jobs with a given status: 'draft', 'open' or 'closed'.
        Returns id, title, location and remote for each job."""
        async with session_factory() as session:
            jobs = await job_service.list_jobs(session, status)
        return [
            {"id": j.id, "title": j.title, "location": j.location, "remote": j.remote} for j in jobs
        ]

    @tool
    async def get_job_details(job_id: int) -> dict:
        """Get the full details of one job by its numeric id: description, requirements, status."""
        async with session_factory() as session:
            try:
                job = await job_service.get_job(session, job_id)
            except JobNotFoundError as e:
                return {"error": str(e)}  # the model reads this and can recover
        return {
            "id": job.id,
            "title": job.title,
            "status": job.status.value,
            "location": job.location,
            "must_have": job.must_have,
            "nice_to_have": job.nice_to_have,
            "description": job.description,
        }

    @tool
    async def close_job(job_id: int) -> dict:
        """Close an OPEN job so it no longer accepts applications.
        Only use this when the recruiter clearly asked to close that specific job."""
        async with session_factory() as session:
            try:
                job = await job_service.close_job(session, job_id)
            except (JobNotFoundError, JobNotOpenError) as e:
                return {"error": str(e)}
        return {"id": job.id, "status": job.status.value}

    @tool
    async def publish_job(job_id: int) -> dict:
        """Publish a DRAFT job so it becomes open and accepts applications. 
        Only use this when the recruiter clearly asks to publish that specific job."""
        async with session_factory() as session:
            try:
                job = await job_service.publish_job(session, job_id)
            except (JobNotFoundError, JobNotDraftError) as e:
                return {"error": str(e)}
        return {"id": job.id, "status": job.status.value}

    return [list_jobs, get_job_details, close_job, publish_job]