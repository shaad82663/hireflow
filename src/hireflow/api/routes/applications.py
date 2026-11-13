from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from hireflow.agents.resume_qa import AnswerWithSources, answer_about_applicants
from hireflow.api.deps import SessionDep, ModelDep, EmbeddingsDep
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

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


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

@router.post("/{job_id}/ask", response_model=AnswerWithSources)
async def ask_about_applicants(
    job_id: int,
    payload: AskRequest,
    session: SessionDep,
    model: ModelDep,
    embeddings: EmbeddingsDep,
):
    """Ask a question about this job's applicants. The answer cites resume excerpts."""
    await job_service.get_job(session, job_id)
    return await answer_about_applicants(model, session, embeddings, job_id, payload.question)