import json

from fakes import KeywordScorer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select

from hireflow.api.deps import get_checkpointer
from hireflow.core.config import get_settings
from hireflow.db.session import SessionLocal
from hireflow.llm.client import get_chat_model
from hireflow.models.application import Application, ApplicationStage
from hireflow.models.screening import ScreeningScore

# job_payload: must have Python + FastAPI, nice to have AWS
RESUMES = {
    "Asha Rao": "Senior engineer, 6 years. Built Python services with FastAPI, deployed on AWS.",
    "Ravi Kumar": "Data engineer, 4 years. Python and Spark pipelines for retail analytics teams.",
    "Meera Iyer": "Backend developer, 3 years. Uses Python and FastAPI daily for payment APIs.",
}


async def apply_all(client, job) -> dict[str, int]:
    """Apply every resume to the job; return {name: application_id}."""
    for i, (name, text) in enumerate(RESUMES.items()):
        await client.post(
            f"/careers/{job['slug']}/apply",
            data={"name": name, "email": f"c{i}@example.com", "consent": "true"},
            files={"resume": (f"{i}.txt", text.encode(), "text/plain")},
        )
    apps = (await client.get(f"/api/jobs/{job['id']}/applications")).json()
    return {a["candidate_name"]: a["id"] for a in apps}


async def test_screening_pauses_with_a_ranked_proposal(app, client, open_job):
    ids = await apply_all(client, open_job)
    scorer = KeywordScorer()
    app.dependency_overrides[get_chat_model] = lambda: scorer

    res = await client.post(f"/api/jobs/{open_job['id']}/screenings", json={"min_score": 60})

    assert res.status_code == 201
    body = res.json()
    assert body["status"] == "awaiting_review"
    assert [r["application_id"] for r in body["ranking"]][0] == ids["Asha Rao"]  # best first
    assert body["proposal"] == [ids["Asha Rao"], ids["Meera Iyer"]]  # Ravi misses a must-have
    assert len(scorer.prompts) == 3  # one parallel scoring call per applicant
    assert all("Asha" not in p and "Ravi" not in p for p in scorer.prompts)  # anonymised


async def test_decision_updates_stages_and_writes_audit_rows(app, client, open_job):
    ids = await apply_all(client, open_job)
    app.dependency_overrides[get_chat_model] = KeywordScorer
    run = (await client.post(f"/api/jobs/{open_job['id']}/screenings", json={})).json()

    # The recruiter overrides the AI: shortlists only Asha, rejects Ravi, leaves Meera undecided.
    decision = {"shortlist": [ids["Asha Rao"]], "reject": [ids["Ravi Kumar"]], "note": "Asha first"}
    res = await client.post(f"/api/screenings/{run['thread_id']}/decision", json=decision)

    assert res.json()["status"] == "completed"
    async with SessionLocal() as session:
        stages = dict((await session.execute(select(Application.id, Application.stage))).all())
        audit = (await session.execute(select(ScreeningScore))).scalars().all()
    assert stages[ids["Asha Rao"]] == ApplicationStage.SHORTLISTED
    assert stages[ids["Ravi Kumar"]] == ApplicationStage.REJECTED
    assert stages[ids["Meera Iyer"]] == ApplicationStage.APPLIED
    assert {a.decision.value for a in audit} == {"shortlisted", "rejected", "undecided"}
    assert all(a.reviewer_note == "Asha first" for a in audit)


async def test_decision_is_validated(app, client, open_job):
    ids = await apply_all(client, open_job)
    app.dependency_overrides[get_chat_model] = KeywordScorer
    run = (await client.post(f"/api/jobs/{open_job['id']}/screenings", json={})).json()
    url = f"/api/screenings/{run['thread_id']}/decision"
    asha = ids["Asha Rao"]

    assert (await client.post(url, json={"shortlist": [99999]})).status_code == 422
    assert (await client.post(url, json={"shortlist": [asha], "reject": [asha]})).status_code == 422
    assert (await client.post(url, json={"shortlist": [asha]})).status_code == 200
    assert (await client.post(url, json={"shortlist": [asha]})).status_code == 409  # already done


async def test_no_applicants_finishes_without_review(app, client, open_job):
    app.dependency_overrides[get_chat_model] = KeywordScorer
    res = await client.post(f"/api/jobs/{open_job['id']}/screenings", json={})
    assert res.json()["status"] == "no_applicants"


async def test_paused_run_survives_a_restart(app, client, open_job):
    """With the Postgres checkpointer, a NEW connection (think: server restarted) can resume."""
    ids = await apply_all(client, open_job)
    app.dependency_overrides[get_chat_model] = KeywordScorer
    url = get_settings().checkpoint_db_url

    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        app.dependency_overrides[get_checkpointer] = lambda: saver
        run = (await client.post(f"/api/jobs/{open_job['id']}/screenings", json={})).json()

    async with AsyncPostgresSaver.from_conn_string(url) as fresh_saver:  # "after the restart"
        app.dependency_overrides[get_checkpointer] = lambda: fresh_saver
        status = (await client.get(f"/api/screenings/{run['thread_id']}")).json()
        assert status["status"] == "awaiting_review"
        done = await client.post(
            f"/api/screenings/{run['thread_id']}/decision", json={"shortlist": [ids["Asha Rao"]]}
        )
        assert done.json()["status"] == "completed"


async def test_history_lists_checkpoints(app, client, open_job):
    await apply_all(client, open_job)
    app.dependency_overrides[get_chat_model] = KeywordScorer
    run = (await client.post(f"/api/jobs/{open_job['id']}/screenings", json={})).json()

    steps = (await client.get(f"/api/screenings/{run['thread_id']}/history")).json()

    assert steps[0]["next"] == ["human_review"]  # newest checkpoint: waiting for the human
    assert len(steps) >= 4  # start, load_job, scoring, rank ...


async def test_stream_reports_progress(app, client, open_job):
    await apply_all(client, open_job)
    app.dependency_overrides[get_chat_model] = KeywordScorer

    res = await client.post(f"/api/jobs/{open_job['id']}/screenings/stream", json={})

    events = [
        line.removeprefix("event: ") for line in res.text.splitlines() if line.startswith("event:")
    ]
    assert events[:2] == ["started", "loaded"]
    assert events.count("scored") == 3
    assert events[-2:] == ["awaiting_review", "done"]
    first_data = json.loads(res.text.split("data: ", 1)[1].split("\n", 1)[0])
    assert first_data["thread_id"].startswith(f"job{open_job['id']}-")