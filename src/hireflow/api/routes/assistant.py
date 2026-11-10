from fastapi import APIRouter
from pydantic import BaseModel, Field

from hireflow.agents.assistant import AssistantReply, ask
from hireflow.agents.tools import build_job_tools
from hireflow.api.deps import ModelDep
from hireflow.db.session import SessionLocal

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=2000)


@router.post("/chat", response_model=AssistantReply)
async def chat(payload: ChatRequest, model: ModelDep):
    return await ask(model, build_job_tools(SessionLocal), payload.message)