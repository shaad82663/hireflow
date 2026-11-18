import hashlib
import secrets
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from hireflow.api.deps import SessionDep, SettingsDep
from hireflow.integrations.email import latest_reply, token_from_address
from hireflow.models.interview import Interview
from hireflow.services import interviews as interview_service
from hireflow.services.tasks import enqueue

router = APIRouter(prefix="/api", tags=["interviews"])


class ScheduleRequest(BaseModel):
    interviewer_email: EmailStr


class InboundEmail(BaseModel):
    """The shape our webhook accepts. Email providers (SendGrid, Postmark, SES) can forward
    incoming mail to a URL; a few lines of mapping turn their format into this one."""

    from_email: str
    to: str
    subject: str = ""
    text: str = Field(max_length=50_000)
    message_id: str | None = None


@router.post("/jobs/{job_id}/interviews", status_code=status.HTTP_202_ACCEPTED)
async def schedule_interviews(job_id: int, payload: ScheduleRequest, session: SessionDep):
    """Start scheduling for every shortlisted candidate. Emails go out from the worker."""
    created = await interview_service.start_scheduling_for_job(
        session, job_id, payload.interviewer_email
    )
    return {"started": len(created), "interview_ids": [i.id for i in created]}


@router.get("/jobs/{job_id}/interviews")
async def list_interviews(job_id: int, session: SessionDep) -> list[dict]:
    return await interview_service.list_interviews(session, job_id)


@router.get("/interviews/{interview_id}")
async def get_interview(interview_id: int, session: SessionDep) -> dict:
    interview = await session.get(Interview, interview_id)
    if interview is None:
        raise HTTPException(404, f"Interview {interview_id} not found")
    return {
        "id": interview.id,
        "status": interview.status.value,
        "starts_at": interview.starts_at.isoformat() if interview.starts_at else None,
        "ends_at": interview.ends_at.isoformat() if interview.ends_at else None,
        "kit": interview.kit,
    }


@router.post("/inbound-email", status_code=status.HTTP_202_ACCEPTED)
async def inbound_email(
    payload: InboundEmail,
    session: SessionDep,
    settings: SettingsDep,
    x_inbound_secret: Annotated[str, Header()] = "",
):
    """Webhook for candidate replies. It only QUEUES the work and answers fast (202):
    email providers retry webhooks that are slow or fail."""
    expected = settings.inbound_secret.get_secret_value()
    if not secrets.compare_digest(x_inbound_secret, expected):  # constant-time comparison
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid inbound secret")

    token = token_from_address(payload.to)
    interview = await interview_service.find_by_reply_token(session, token) if token else None
    if interview is None:
        raise HTTPException(404, "No interview matches this reply address")

    reply = latest_reply(payload.text)
    # The same email delivered twice must create only one task.
    key_source = payload.message_id or f"{payload.to}|{reply}"
    key = "reply-" + hashlib.sha256(key_source.encode()).hexdigest()[:32]
    task_id = await enqueue(
        session,
        "resume_scheduling",
        {"interview_id": interview.id, "reply": reply},
        idempotency_key=key,
    )
    await session.commit()
    return {"queued": task_id is not None}