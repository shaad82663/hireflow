from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from hireflow.api.deps import SessionDep
from hireflow.models.application import ApplicationStage
from hireflow.services import applications as  app_service
from hireflow.services import jobs as job_service

router = APIRouter(prefix="/api/jobs", tags=["applications"])

class ApplicationRead(BaseModel):
    id: int
    candidate_name: str
    candidate_email: str
    resume_filename: str
    stage: ApplicationStage
    created_at: datetime


@router.get("/{job_id}/applications", response_model=list[ApplicationRead])
async def list_applications(job_id: int, session: SessionDep):
    await job_service.get_job(session, job_id)  # 404 if the job does not exist
    apps = await app_service.list_applications(session, job_id)
    return [
        ApplicationRead(
            id=a.id,
            candidate_name=a.candidate.name,
            candidate_email=a.candidate.email,
            resume_filename=a.resume_filename,
            stage=a.stage,
            created_at=a.created_at,
        )
        for a in apps
    ]