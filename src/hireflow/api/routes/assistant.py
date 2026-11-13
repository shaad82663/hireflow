from fastapi import APIRouter
from pydantic import BaseModel, Field

from hireflow.agents.assistant import AssistantReply, ask_assistant
from hireflow.agents.resume_tools import build_resume_tools
from hireflow.agents.tools import build_job_tools
from hireflow.api.deps import EmbeddingsDep, ModelDep
from hireflow.db.session import SessionLocal

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=2000)


@router.post("/chat", response_model=AssistantReply)
async def chat(payload: ChatRequest, model: ModelDep, embeddings: EmbeddingsDep):
    tools = build_job_tools(SessionLocal) + build_resume_tools(SessionLocal, embeddings)
    return await ask_assistant(model, tools, payload.message)