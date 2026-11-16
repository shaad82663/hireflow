from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver

from fastapi import Depends, Request
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