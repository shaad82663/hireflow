from fastapi import APIRouter
from sqlalchemy import text

from hireflow.api.deps import SessionDep, SettingsDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: SettingsDep) -> dict[str,str]:
    return {"status": "ok", "app": settings.app_name, "env": settings.environment  }


@router.get("/health/db")
async def health_db(session: SessionDep) -> dict[str,str]:
    await session.execute(text("SELECT 1"))
    return {"database": "ok"}


