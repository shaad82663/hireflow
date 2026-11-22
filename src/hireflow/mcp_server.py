import logging
import sys

from langchain_core.embeddings import Embeddings
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hireflow.models.job import JobStatus
from hireflow.rag.search import hybrid_search
from hireflow.services import interviews as interview_service
from hireflow.services import jobs as job_service
from hireflow.services.jobs import JobNotFoundError


log = logging.getLogger("hireflow.mcp")

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

def _job_dict(job) -> dict:
    return {
        "id": job.id,
        "title": job.title,
        "status": job.status.value,
        "location": job.location,
        "remote": job.remote,
        "must_have": job.must_have,
        "nice_to_have": job.nice_to_have,
        "description": job.description,
    }


def build_server(
        session_factory: async_sessionmaker[AsyncSession],
        embeddings: Embeddings,
        *,
        allow_writes: bool = False,
) -> FastMCP:
    
    mcp = FastMCP(
        "hireflow",
        instructions=(
            "HireFlow is a hiring system. Use these tools to look up jobs, search applicant "
            "resumes and check interview status. Job ids are numbers from list_open_jobs."
        ),
    )

    @mcp.tool(annotations=READ_ONLY)
    async def list_open_jobs() -> list[dict]:
        """List the jobs that are currently open, with id, title and location."""
        log.info("tool list_open_jobs")
        async with session_factory() as session:
            jobs = await job_service.list_jobs(session, JobStatus.OPEN)
        return [{"id": j.id, "title": j.title, "location": j.location} for j in jobs]

    @mcp.tool(annotations=READ_ONLY)
    async def get_job(job_id: int) -> dict:
        """Full details of one job: description, must-have and nice-to-have skills, status."""
        log.info("tool get_job job_id=%s", job_id)
        async with session_factory() as session:
            try:
                return _job_dict(await job_service.get_job(session, job_id))
            except JobNotFoundError as e:
                raise ToolError(str(e)) from e 

    @mcp.tool(annotations=READ_ONLY)
    async def search_resumes(job_id: int, query: str, limit: int = 5) -> list[dict]:
        """Search the resumes of one job's applicants (hybrid keyword + meaning search).
        Use a specific query such as 'Kubernetes in production'."""
        log.info("tool search_resumes job_id=%s", job_id)  
        if not 1 <= limit <= 10:
            raise ToolError("limit must be between 1 and 10")
        async with session_factory() as session:
            hits = await hybrid_search(session, embeddings, job_id, query, k=limit)
        return [
            {
                "application_id": h.application_id,
                "candidate": h.candidate_name,
                "excerpt": h.content,
            }
            for h in hits
        ]

    @mcp.tool(annotations=READ_ONLY)
    async def interview_status(job_id: int) -> list[dict]:
        """Interviews for one job: candidate, status (proposed/booked/...) and start time."""
        log.info("tool interview_status job_id=%s", job_id)
        async with session_factory() as session:
            return await interview_service.list_interviews(session, job_id)

    @mcp.resource("hireflow://jobs/{job_id}", mime_type="application/json")
    async def job_resource(job_id: int) -> dict:
        """One job as a document the user can attach to a chat."""
        async with session_factory() as session:
            return _job_dict(await job_service.get_job(session, int(job_id)))

    @mcp.prompt()
    def screening_brief(job_id: int, shortlist_zone: int = 2) -> str:
        """A ready-made prompt: summarize the strongest applicants for a job."""
        return (
            f"Use get_job to read job {job_id}. Then use search_resumes with each must-have "
            "skill. Write a short brief: the {shortlist_zone} strongest applicants, the evidence for each, and "
            "what to ask them in an interview. Judge only job-related skills and experience."
        )

    if allow_writes:

        @mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True))
        async def close_job(job_id: int) -> dict:
            """Close an open job so it stops accepting applications. Ask the user first."""
            log.warning("tool close_job job_id=%s", job_id)  # changes are logged louder
            async with session_factory() as session:
                try:
                    job = await job_service.close_job(session, job_id)
                except (JobNotFoundError, ValueError) as e:
                    raise ToolError(str(e)) from e
            return {"id": job.id, "status": job.status.value}

    return mcp


def main() -> None:
    from hireflow.core.config import get_settings
    from hireflow.db.session import SessionLocal
    from hireflow.rag.embeddings import get_embeddings

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(message)s")
    settings = get_settings()
    server = build_server(SessionLocal, get_embeddings(), allow_writes=settings.mcp_allow_writes)
    server.run("stdio")


if __name__ == "__main__":
    main()

    