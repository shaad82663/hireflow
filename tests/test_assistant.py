from fakes import FakeToolModel, tool_call
from langchain_core.messages import AIMessage

from hireflow.agents.tools import build_job_tools
from hireflow.db.session import SessionLocal
from hireflow.llm.client import get_chat_model


async def test_list_jobs_tool_reads_the_database(open_job):
    list_jobs, _, _ = build_job_tools(SessionLocal)
    result = await list_jobs.ainvoke({"status": "open"})
    assert result == [
        {
            "id": open_job["id"],
            "title": "Senior Python Engineer",
            "location": "Bengaluru",
            "remote": True,
        }
    ]


async def test_tool_errors_come_back_as_data(client, job_payload):
    draft = (await client.post("/api/jobs", json=job_payload)).json()
    _, get_job_details, close_job = build_job_tools(SessionLocal)

    assert await get_job_details.ainvoke({"job_id": 999}) == {"error": "Job 999 not found"}
    assert await close_job.ainvoke({"job_id": draft["id"]}) == {
        "error": "Only open jobs can be closed"
    }


async def test_assistant_endpoint_runs_the_agent_loop(app, client, open_job):
    script = [
        tool_call("list_jobs", {"status": "open"}),
        AIMessage("You have 1 open job: Senior Python Engineer in Bengaluru."),
    ]
    app.dependency_overrides[get_chat_model] = lambda: FakeToolModel(messages=iter(script))

    res = await client.post("/api/assistant/chat", json={"message": "What jobs are open?"})

    assert res.status_code == 200
    assert res.json() == {
        "reply": "You have 1 open job: Senior Python Engineer in Bengaluru.",
        "tools_used": ["list_jobs"],
    }


async def test_assistant_can_close_a_job_when_asked(app, client, open_job):
    script = [
        tool_call("close_job", {"job_id": open_job["id"]}),
        AIMessage("Done. The job is now closed."),
    ]
    app.dependency_overrides[get_chat_model] = lambda: FakeToolModel(messages=iter(script))

    await client.post("/api/assistant/chat", json={"message": f"Close job {open_job['id']}"})

    job = (await client.get(f"/api/jobs/{open_job['id']}")).json()
    assert job["status"] == "closed"  # the tool really changed the database