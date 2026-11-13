from langchain_core.embeddings import Embeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload


from hireflow.models.application import Application, Candidate, ResumeChunk
from hireflow.models.job import Job
from hireflow.rag.chunking import chunk_text
from hireflow.services.resume_parser import extract_text


class DuplicateApplicationError(Exception):
    def __init__(self) -> None:
        super().__init__("You have already applied to this job. We will be in touch.")

class ConsentRequiredError(Exception):
    def __init__(self) -> None:
        super().__init__("Please agree to the privacy notice so we can process your resume.")


async def apply_to_job(
        session:  AsyncSession,
        embeddings: Embeddings,
        job: Job,
        *,
        name: str,
        email: str,
        filename: str,
        data: bytes,
        consent: bool
) -> Application:
        """Parse the resume, save the application, and index it for search, in one transaction."""
        if not consent:
             raise ConsentRequiredError()
        text = extract_text(filename, data)
        email = email.strip().lower()
        stmt = select(Candidate).where(Candidate.email == email)
        candidate = (await session.execute(stmt)).scalar_one_or_none()
        if candidate is None:
             candidate = Candidate(name=name.strip(), email=email)
             session.add(candidate)
             await session.flush()
        else:
            already = await session.execute(
                select(Application.id).where(
                    Application.job_id == job.id, Application.candidate_id == candidate.id
                )
            )
            if already.first() is not None:
                raise DuplicateApplicationError()

        application = Application(
            job_id = job.id,
            candidate_id = candidate.id,
            resume_filename=filename,
            resume_text=text,
            consent_given=True
        )
        session.add(application)

        chunks = chunk_text(text)
        vectors = await embeddings.aembed_documents(chunks)
        application.chunks = [
            ResumeChunk(chunk_index=i, content=c, embedding=v)
            for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
        ]

        await session.commit()
        await session.refresh(application, attribute_names=["candidate"])
        return application


async def list_applications(session: AsyncSession, job_id: int) -> list[Application]:
     stmt = (
          select(Application)
          .where(Application.job_id == job_id)
          .options(selectinload(Application.candidate))
          .order_by(Application.created_at.desc(), Application.id.desc())
     )
     return list((await session.execute(stmt)).scalars().all())


