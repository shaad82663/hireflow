from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore  

from fastapi import Depends, Header, Request 
from sqlalchemy.ext.asyncio import AsyncSession

from hireflow.core.config import get_settings, Settings
from hireflow.db.session import get_session
from hireflow.llm.client import get_chat_model
from hireflow.rag.embeddings import get_embeddings

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ModelDep = Annotated[BaseChatModel, Depends(get_chat_model)]
EmbeddingsDep = Annotated[Embeddings, Depends(get_embeddings)]


def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    return request.app.state.checkpointer

CheckpointerDep = Annotated[BaseCheckpointSaver, Depends(get_checkpointer)]

def get_store(request: Request) -> BaseStore:
    """The long-term memory store, created once at startup next to the checkpointer."""
    return request.app.state.store


def get_recruiter_id(
    x_recruiter_id: Annotated[str, Header(pattern=r"^[a-z0-9_-]{1,40}$")] = "demo",
) -> str:
    """Who is using the app. A simple header until Part 13 adds real login."""
    return x_recruiter_id

StoreDep = Annotated[BaseStore, Depends(get_store)]
RecruiterDep = Annotated[str, Depends(get_recruiter_id)]