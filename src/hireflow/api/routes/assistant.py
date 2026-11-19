from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from hireflow.agents.assistant import AssistantReply, ask_assistant
from hireflow.agents.coordinator_tools import build_interview_tools
from hireflow.agents.memory_tools import build_memory_tools
from hireflow.agents.resume_tools import build_resume_tools
from hireflow.agents.tools import build_job_tools
from hireflow.api.deps import CheckpointerDep, EmbeddingsDep, ModelDep, RecruiterDep, StoreDep
from hireflow.db.session import SessionLocal
from hireflow.memory.preferences import list_preferences, preferences_prompt

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=2000)
    thread_id: str | None = Field(
        default=None, description="Leave empty to start a new conversation"
    )


def check_owner(thread_id: str, recruiter: str) -> None:
    """Thread ids start with the recruiter's id, so nobody can read another person's chat."""
    if not thread_id.startswith(f"chat-{recruiter}-"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This conversation belongs to someone else")


@router.post("/chat", response_model=AssistantReply)
async def chat(
    payload: ChatRequest,
    model: ModelDep,
    embeddings: EmbeddingsDep,
    checkpointer: CheckpointerDep,
    store: StoreDep,
    recruiter: RecruiterDep,
):
    thread_id = payload.thread_id or f"chat-{recruiter}-{uuid4().hex[:12]}"
    check_owner(thread_id, recruiter)

    tools = (
        build_job_tools(SessionLocal)
        + build_resume_tools(SessionLocal, embeddings)
        + build_interview_tools(SessionLocal)  # hands work to the scheduling agent
        + build_memory_tools(store, recruiter)  # long-term memory for this recruiter only
    )
    preferences = await list_preferences(store, recruiter)
    return await ask_assistant(
        model,
        tools,
        payload.message,
        thread_id=thread_id,
        checkpointer=checkpointer,
        extra_system=preferences_prompt(preferences),
    )


@router.get("/threads/{thread_id}")
async def read_thread(thread_id: str, checkpointer: CheckpointerDep, recruiter: RecruiterDep):
    """The saved messages of one conversation (what the agent 'remembers')."""
    check_owner(thread_id, recruiter)
    checkpoint = await checkpointer.aget({"configurable": {"thread_id": thread_id}})
    if checkpoint is None:
        raise HTTPException(404, "Conversation not found")
    return [
        {"role": m.type, "text": m.text}
        for m in checkpoint["channel_values"].get("messages", [])
        if m.type in ("human", "ai") and m.text  # skip tool messages and empty tool-call turns
    ]