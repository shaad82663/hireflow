from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from hireflow.api.routes import applications, assistant, drafts, health, jobs
from hireflow.core.config import get_settings
from hireflow.core.logging import setup_logging
from hireflow.llm.client import MissingAPIKeyError
from hireflow.services.jobs import JobNotFoundError
from hireflow.web import routes as web_routes


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
    app.include_router(web_routes.router)
    return app


app = create_app()