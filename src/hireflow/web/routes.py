import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from hireflow.api.deps import SessionDep, SettingsDep
from hireflow.models.job import Job, JobStatus
from hireflow.services import jobs as job_service

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(tags=["careers"], include_in_schema=False)


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