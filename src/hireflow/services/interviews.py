import secrets
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from hireflow.models.application import Application, ApplicationStage
from hireflow.models.interview import Interview
from hireflow.services import jobs as job_service
from hireflow.services.tasks import enqueue

async def start_scheduling_for_job(
    session: AsyncSession, job_id: int, interviewer_email: str
) -> list[Interview]:
    """Create an Interview for every shortlisted application that has none yet,
    and queue one 'start_scheduling' task each. Safe to call twice."""
    await job_service.get_job(session, job_id)
    already = select(Interview.application_id)
    stmt = (
        select(Application)
        .where(
            Application.job_id == job_id,
            Application.stage == ApplicationStage.SHORTLISTED,
            Application.id.not_in(already),
        )
        .order_by(Application.id)
    )
    created = []
    for app in (await session.execute(stmt)).scalars().all():
        interview = Interview(
            application_id=app.id,
            interviewer_email=interviewer_email.strip().lower(),
            thread_id=f"interview-{uuid4().hex[:12]}",
            reply_token=secrets.token_urlsafe(9),  # unguessable: part of the reply address
        )
        session.add(interview)
        await session.flush()  # get interview.id
        await enqueue(
            session,
            "start_scheduling",
            {"interview_id": interview.id},
            idempotency_key=f"start-interview-{interview.id}",
        )
        created.append(interview)
    await session.commit()  # interviews and their tasks are saved together
    return created

async def list_interviews(session: AsyncSession, job_id: int) -> list[dict]:
    stmt = (
        select(Interview, Application)
        .join(Application, Interview.application_id == Application.id)
        .where(Application.job_id == job_id)
        .options(selectinload(Application.candidate))
        .order_by(Interview.id)
    )
    return [
        {
            "interview_id": iv.id,
            "application_id": app.id,
            "candidate": app.candidate.name,
            "status": iv.status.value,
            "starts_at": iv.starts_at.isoformat() if iv.starts_at else None,
            "has_kit": iv.kit is not None,
        }
        for iv, app in (await session.execute(stmt)).all()
    ]


async def find_by_reply_token(session: AsyncSession, token: str) -> Interview | None:
    stmt = select(Interview).where(Interview.reply_token == token)
    return (await session.execute(stmt)).scalar_one_or_none()