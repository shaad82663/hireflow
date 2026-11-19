from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore 

from hireflow.api.routes import (
    applications,
    assistant,
    drafts,
    health,
    interviews,
    jobs,
    memories,
    screenings,
)
from hireflow.core.config import get_settings
from hireflow.core.logging import setup_logging
from hireflow.llm.client import MissingAPIKeyError
from hireflow.services.jobs import JobNotFoundError
from hireflow.web import routes as web_routes

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup (code before yield) and once at shutdown (code after yield)."""
    settings = get_settings()
    url = settings.checkpoint_db_url
    async with (
        AsyncPostgresSaver.from_conn_string(url) as checkpointer,  # short-term: graph state
        AsyncPostgresStore.from_conn_string(url) as store,  # long-term: memories
    ):
        await checkpointer.setup()  # create LangGraph's own tables if missing (safe to repeat)
        await store.setup()
        app.state.checkpointer = checkpointer
        app.state.store = store
        yield


def create_app() -> FastAPI:
    """App factory: builds a fresh FastAPI app (tests call this too)."""
    settings = get_settings()
    setup_logging(settings.debug)

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    @app.exception_handler(JobNotFoundError)
    async def job_not_found(request: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(MissingAPIKeyError)
    async def missing_key(request: Request, exc: MissingAPIKeyError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(drafts.router)
    app.include_router(applications.router)
    app.include_router(assistant.router)
    app.include_router(screenings.router)
    app.include_router(interviews.router)
    app.include_router(memories.router)
    app.include_router(web_routes.router)
    return app


app = create_app()