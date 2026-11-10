from typing import Annotated

from langchain_core.language_models import BaseChatModel

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from hireflow.core.config import get_settings, Settings
from hireflow.db.session import get_session
from hireflow.llm.client import get_chat_model

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ModelDep = Annotated[BaseChatModel, Depends(get_chat_model)]