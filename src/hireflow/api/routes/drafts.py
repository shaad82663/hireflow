from fastapi import APIRouter
from pydantic import BaseModel, Field

from hireflow.agents.intake import JobDraft, draft_job
from hireflow.api.deps import ModelDep, SettingsDep

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class DraftRequest(BaseModel):
    notes: str = Field(min_length=20, max_length=5000)


@router.post("/draft", response_model=JobDraft)
async def create_draft(payload: DraftRequest, model: ModelDep, settings: SettingsDep):
    """Rough notes in, structured job post out. A human reviews it before saving."""
    return await draft_job(model, payload.notes, settings.company_name)