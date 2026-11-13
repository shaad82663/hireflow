from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from hireflow.api.routes import assistant, drafts, health, jobs, applications
from hireflow.core.config import get_settings
from hireflow.core.logging import setup_logging
from hireflow.services.jobs import JobNotFoundError
from hireflow.web import routes as web_routes


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.debug)

    app = FastAPI(title=settings.app_name, version="1.0.0")

    @app.exception_handler(JobNotFoundError)
    async def job_not_found(request: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(web_routes.router)
    app.include_router(drafts.router)
    app.include_router(assistant.router)
    app.include_router(applications.router)
    return app

app = create_app()