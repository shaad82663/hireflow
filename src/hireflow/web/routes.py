import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import EmailStr, TypeAdapter, ValidationError

from hireflow.api.deps import EmbeddingsDep, SessionDep, SettingsDep
from hireflow.models.job import Job, JobStatus
from hireflow.services import applications as app_service
from hireflow.services import jobs as job_service
from hireflow.services.applications import ConsentRequiredError, DuplicateApplicationError
from hireflow.services.resume_parser import ResumeParseError

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(tags=["careers"], include_in_schema=False)

_email_check = TypeAdapter(EmailStr)

def job_posting_jsonld(job: Job, company_name: str, company_url: str) -> str:
    """Google for Jobs reads this block to list your job in Google Search for free."""
    data = {
        "@context": "https://schema.org/",
        "@type": "JobPosting",
        "title": job.title,
        "description": job.description,
        "datePosted": job.created_at.date().isoformat(),
        "employmentType": job.employment_type.value,
        "hiringOrganization": {
            "@type": "Organization",
            "name": company_name,
            "sameAs": company_url,
        },
        "jobLocation": {
            "@type": "Place",
            "address": {"@type": "PostalAddress", "addressLocality": job.location},
        },
    }
    if job.remote:
        data["jobLocationType"] = "TELECOMMUTE"
    # escape "</" so a job description can never close the <script> tag early
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

def _detail_page(request: Request, job: Job, settings, status_code: int = 200, **extra):
    jsonld = job_posting_jsonld(
        job, settings.company_name, settings.company_url
    )
    context = {"job": job, "company": settings.company_name, "jsonld": jsonld, **extra}
    return templates.TemplateResponse(request, "job_detail.html", context, status_code=status_code)

@router.get("/careers", response_class=HTMLResponse)
async def careers_list(request: Request, session: SessionDep, settings: SettingsDep):
    jobs = await job_service.list_jobs(session, status=JobStatus.OPEN)
    return templates.TemplateResponse(
        request, "careers_list.html", {"jobs": jobs, "company": settings.company_name}
    )


@router.get("/careers/{slug}", response_class=HTMLResponse)
async def careers_detail(slug: str, request: Request, session: SessionDep, settings: SettingsDep):
    job = await job_service.get_open_job_by_slug(session, slug)
    jsonld = job_posting_jsonld(job, settings.company_name, settings.company_url)
    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {"job": job, "company": settings.company_name, "jsonld": jsonld},
    )

@router.post("/careers/{slug}/apply", response_class=HTMLResponse)
async def apply(
    slug: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    embeddings: EmbeddingsDep,
    name: Annotated[str, Form(min_length=2, max_length=120)],
    email: Annotated[str, Form(max_length=254)],
    resume: UploadFile,
    consent: Annotated[bool, Form()] = False,  # an unticked checkbox sends nothing at all
):
    job = await job_service.get_open_job_by_slug(session, slug)
    form = {"name": name, "email": email}

    try:
        _email_check.validate_python(email)
    except ValidationError:
        return _detail_page(
            request, job, settings, 422, error="Please enter a valid email.", form=form
        )

    max_bytes = settings.max_resume_mb * 1024 * 1024
    data = await resume.read(max_bytes + 1)  # read at most one byte more than allowed
    if len(data) > max_bytes:
        error = f"Resume is larger than {settings.max_resume_mb} MB."
        return _detail_page(request, job, settings, 413, error=error, form=form)

    try:
        await app_service.apply_to_job(
            session,
            embeddings,
            job,
            name=name,
            email=email,
            filename=resume.filename or "resume",
            data=data,
            consent=consent,
        )
    except DuplicateApplicationError as e:
        return _detail_page(request, job, settings, 409, error=str(e), form=form)
    except (ResumeParseError, ConsentRequiredError) as e:
        return _detail_page(request, job, settings, 422, error=str(e), form=form)

    return templates.TemplateResponse(
        request, "applied.html", {"job": job, "company": settings.company_name, "name": name}
    )
