from sqlalchemy import func, select

from hireflow.db.session import SessionLocal
from hireflow.models.application import Application, ResumeChunk

RESUME = (
    "Asha Rao - Backend Engineer\n\n"
    "5 years building Python APIs with FastAPI and PostgreSQL. "
    "Deployed services on AWS with Docker and Kubernetes."
)


def form(**overrides) -> dict:
    data = {"name": "Asha Rao", "email": "Asha@Example.com", "consent": "true"}
    data.update(overrides)
    return data


def resume_file(text: str = RESUME, name: str = "asha.txt") -> dict:
    return {"resume": (name, text.encode(), "text/plain")}


async def test_apply_saves_application_and_chunks(client, open_job):
    res = await client.post(f"/careers/{open_job['slug']}/apply", data=form(), files=resume_file())

    assert res.status_code == 200
    assert "Thanks, Asha Rao!" in res.text
    async with SessionLocal() as session:
        app = (await session.execute(select(Application))).scalar_one()
        chunks = (await session.execute(select(func.count(ResumeChunk.id)))).scalar_one()
    assert app.resume_text.startswith("Asha Rao")
    assert chunks >= 1


async def test_duplicate_application_is_rejected(client, open_job):
    url = f"/careers/{open_job['slug']}/apply"
    await client.post(url, data=form(), files=resume_file())
    res = await client.post(url, data=form(email="asha@example.com"), files=resume_file())

    assert res.status_code == 409
    assert "already applied" in res.text


async def test_consent_and_email_are_required(client, open_job):
    url = f"/careers/{open_job['slug']}/apply"
    no_consent = {k: v for k, v in form().items() if k != "consent"}
    assert (await client.post(url, data=no_consent, files=resume_file())).status_code == 422
    assert (await client.post(url, data=form(email="nope"), files=resume_file())).status_code == 422


async def test_bad_file_shows_friendly_error(client, open_job):
    res = await client.post(
        f"/careers/{open_job['slug']}/apply", data=form(), files=resume_file(name="cv.exe")
    )
    assert res.status_code == 422
    assert "Please upload a PDF, DOCX or TXT file." in res.text
    assert 'value="Asha Rao"' in res.text  # the form keeps what the candidate typed


async def test_list_applications(client, open_job):
    await client.post(f"/careers/{open_job['slug']}/apply", data=form(), files=resume_file())
    res = await client.get(f"/api/jobs/{open_job['id']}/applications")

    assert res.status_code == 200
    [row] = res.json()
    assert row["candidate_email"] == "asha@example.com"  # stored lowercase
    assert row["stage"] == "applied"