from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from hireflow.api.routes import applications, assistant, drafts, health, jobs, screenings 
from hireflow.core.config import get_settings
from hireflow.core.logging import setup_logging
from hireflow.llm.client import MissingAPIKeyError
from hireflow.services.jobs import JobNotFoundError
from hireflow.web import routes as web_routes

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup (code before yield) and once at shutdown (code after yield)."""
    settings = get_settings()
    async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_db_url) as checkpointer:
        await checkpointer.setup()  # creates LangGraph's own tables if missing
        app.state.checkpointer = checkpointer
        yield

def create_app() -> FastAPI:
    """App factory: builds a fresh FastAPI app (tests call this too)."""
    settings = get_settings()
    setup_logging(settings.debug)

    app = FastAPI(title=settings.app_name, version="0.1.0")

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
    app.include_router(screenings.router)  # fix: register screening routes
    app.include_router(web_routes.router)
    return app


app = create_app()