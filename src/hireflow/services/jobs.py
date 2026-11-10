import re
import secrets

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hireflow.models.job import Job, JobStatus
from hireflow.schemas.job import JobCreate, JobUpdate


class JobNotFoundError(Exception):
    def __init__(self, job_id: int | str) -> None:
        super().__init__(f"Job {job_id} not found")
        self.job_id = job_id

class JobNotOpenError(Exception):
    def __init__(self, job_id: int) -> None:
        super().__init__("Only open jobs can be closed")
        self.job_id = job_id

class JobNotDraftError(Exception):
    def __init__(self, job_id: int) -> None:
        super().__init__(f"Job {job_id} is not drafted, so it cannot be opened")
        self.job_id = job_id

def slugify(text: str) -> str:
    """'Senior Python Engineer!' -> 'senior-python-engineer'"""
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return text.strip("-")

async def create_job(session: AsyncSession, data: JobCreate) -> Job:
     slug = f"{slugify(data.title)}-{secrets.token_hex(3)}"
     job = Job(**data.model_dump(), slug = slug)
     session.add(job)
     await session.commit()
     await session.refresh(job)
     return job

async def get_job(session: AsyncSession, job_id: int) -> Job:
    job = await session.get(Job, job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    return job

async def get_open_job_by_slug(session: AsyncSession, slug: str) -> Job:
    stmt = select(Job).where(Job.slug == slug, Job.status == JobStatus.OPEN)
    job =(await session.execute(stmt)).scalar_one_or_none()
    if job is None:
        raise JobNotFoundError(slug)
    return job

async def list_jobs(session: AsyncSession, status: JobStatus | None = None) -> list[Job]:
    stmt = select(Job).order_by(Job.created_at.desc(), Job.id.desc())
    if status is not None:
        stmt = stmt.where(Job.status == status)
    result = await session.execute(stmt)
    return list(result.scalars().all())

async def update_job(session: AsyncSession, job_id: int, data: JobUpdate) -> Job:
    job = await get_job(session, job_id)
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(job, field, value)
    await session.commit()
    await session.refresh(job)
    return job

async def delete_job(session: AsyncSession, job_id: int) -> None:
    job = await get_job(session, job_id)
    await session.delete(job)
    await session.commit()


async def close_job(session: AsyncSession, job_id: int) -> Job:
    job = await get_job(session, job_id)
    if job.status is not JobStatus.OPEN:
        raise JobNotOpenError(job_id)
    setattr(job, "status", JobStatus.CLOSED)
    await session.commit()
    await session.refresh(job)
    return job

async def publish_job(session: AsyncSession, job_id: int) -> Job:
    job = await get_job(session, job_id)
    if job.status is not JobStatus.DRAFT:
        raise JobNotDraftError(job_id)
    setattr(job, "status", JobStatus.OPEN)
    await session.commit()
    await session.refresh(job)
    return job