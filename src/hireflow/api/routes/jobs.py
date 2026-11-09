from fastapi import APIRouter, status

from hireflow.api.deps import SessionDep
from hireflow.models.job import JobStatus
from hireflow.schemas.job import JobCreate, JobRead, JobUpdate
from hireflow.services import jobs as job_service

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

@router.post("", response_model=JobRead, status_code=status.HTTP_201_CREATED)
async def create_job(payload: JobCreate, session: SessionDep):
    return await job_service.create_job(session, payload)

@router.get("", response_model=list[JobRead])
async def list_jobs(session: SessionDep, status: JobStatus | None = None):
    return await job_service.list_jobs(session, status)


@router.get("/{job_id}", response_model=JobRead)
async def get_job(job_id: int, session: SessionDep):
    return await job_service.get_job(session, job_id)


@router.patch("/{job_id}", response_model=JobRead)
async def update_job(job_id: int, payload: JobUpdate, session: SessionDep):
    return await job_service.update_job(session, job_id, payload)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: int, session: SessionDep) -> None:
    await job_service.delete_job(session, job_id)