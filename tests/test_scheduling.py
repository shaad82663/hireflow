from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fakes import FakeToolModel, ScriptedStructuredModel, tool_call
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from hireflow.agents.interview_kit import InterviewKit, KitQuestion
from hireflow.agents.scheduling import ReplyIntent
from hireflow.db.session import SessionLocal
from hireflow.integrations.email import OutboxEmailSender, token_from_address
from hireflow.llm.client import get_chat_model
from hireflow.models.application import Application, ApplicationStage
from hireflow.models.interview import CalendarBlock, Interview
from hireflow.models.screening import ScreeningDecision, ScreeningScore
from hireflow.scheduling.calendar import DbCalendar
from hireflow.scheduling.handlers import build_handlers
from hireflow.worker import Worker

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 23, 9, 0, tzinfo=IST)  # a Monday morning
SECRET = {"X-Inbound-Secret": "dev-inbound-secret"}
INTERVIEWER = "lead@acme.com"

RESUME = "Backend engineer with six years of Python, FastAPI and PostgreSQL in production."


def kit() -> InterviewKit:
    return InterviewKit(
        focus_areas=["FastAPI depth"],
        questions=[
            KitQuestion(topic=t, question=f"Tell me about {t}.", good_answer_signals="x")
            for t in ("FastAPI", "Python", "Design")
        ],
    )


class Harness:
    """Everything the worker needs, with fakes for the LLM, email and clock."""

    def __init__(self, *answers):
        self.model = ScriptedStructuredModel(*answers)
        self.outbox = OutboxEmailSender()
        self.handlers = build_handlers(
            model=self.model,
            session_factory=SessionLocal,
            email=self.outbox,
            checkpointer=InMemorySaver(),
            clock=lambda: NOW,
        )
        self.worker = Worker(SessionLocal, self.handlers)

    def to(self, address: str) -> list:
        return [e for e in self.outbox.sent if e.to == address]


@pytest.fixture
async def shortlisted(client, open_job) -> dict:
    """One candidate who applied and was shortlisted (with a screening record)."""
    await client.post(
        f"/careers/{open_job['slug']}/apply",
        data={"name": "Asha Rao", "email": "asha@example.com", "consent": "true"},
        files={"resume": ("a.txt", RESUME.encode(), "text/plain")},
    )
    async with SessionLocal() as session:
        app = (await session.execute(select(Application))).scalar_one()
        app.stage = ApplicationStage.SHORTLISTED
        session.add(
            ScreeningScore(
                thread_id="t",
                application_id=app.id,
                total=70,
                must_haves_met=True,
                decision=ScreeningDecision.SHORTLISTED,
                reviewer_note=None,
                details={
                    "summary": "Solid backend.",
                    "criteria": [
                        {"criterion": "Python", "score": 5, "verified": True},
                        {"criterion": "FastAPI", "score": 2, "verified": False},
                    ],
                },
            )
        )
        await session.commit()
    return {"job_id": open_job["id"], "application_id": app.id}


async def start(client, shortlisted) -> int:
    res = await client.post(
        f"/api/jobs/{shortlisted['job_id']}/interviews", json={"interviewer_email": INTERVIEWER}
    )
    assert res.status_code == 202
    return res.json()["interview_ids"][0]


async def reply(client, harness, text: str, message_id: str | None = None):
    invite = harness.to("asha@example.com")[0]
    res = await client.post(
        "/api/inbound-email",
        headers=SECRET,
        json={
            "from_email": "asha@example.com",
            "to": invite.reply_to,
            "text": text,
            "message_id": message_id,
        },
    )
    assert res.status_code == 202
    await harness.worker.run_until_idle()
    return res.json()


async def interview(interview_id: int) -> Interview:
    async with SessionLocal() as session:
        return await session.get(Interview, interview_id)


async def test_invite_reply_book_and_kit(client, shortlisted):
    h = Harness(ReplyIntent(intent="accept", slot_number=2, summary="Takes option 2"), kit())
    interview_id = await start(client, shortlisted)

    await h.worker.run_until_idle()
    [invite] = h.to("asha@example.com")
    assert "2) Tue 24 Nov, 10:30-11:15 (Asia/Kolkata)" in invite.body  # 12h notice -> Tuesday
    assert token_from_address(invite.reply_to)

    await reply(client, h, "Option 2 is great!\n\nOn Mon, Recruiting wrote:\n> old text")

    iv = await interview(interview_id)
    assert iv.status.value == "booked"
    assert iv.starts_at == datetime(2026, 11, 24, 10, 30, tzinfo=IST)
    assert "Option 2 is great!" in h.model.prompts[0] and "old text" not in h.model.prompts[0]
    assert h.to("asha@example.com")[-1].subject.startswith("Confirmed")
    assert h.to(INTERVIEWER)[-1].subject.startswith("New interview")
    async with SessionLocal() as session:
        [block] = (await session.execute(select(CalendarBlock))).scalars().all()
    assert block.email == INTERVIEWER and block.starts_at == iv.starts_at

    # the kit task was queued by the booking and run by the same worker pass
    detail = (await client.get(f"/api/interviews/{interview_id}")).json()
    assert [q["topic"] for q in detail["kit"]["questions"]] == ["FastAPI", "Python", "Design"]
    kit_prompt = h.model.prompts[1]
    assert "WEAK or UNVERIFIED areas: FastAPI" in kit_prompt  # from the screening record


async def test_other_times_three_times_escalates(client, shortlisted):
    other = ReplyIntent(intent="other_times", summary="None of these work")
    h = Harness(other, other.model_copy(), other.model_copy())
    interview_id = await start(client, shortlisted)
    await h.worker.run_until_idle()

    for i in range(3):
        await reply(client, h, f"Sorry, none of these work ({i})")

    offers = [e for e in h.to("asha@example.com") if "pick a time" in e.subject]
    assert len(offers) == 3
    assert offers[0].body != offers[1].body  # each round offers later slots
    assert (await interview(interview_id)).status.value == "needs_human"
    assert h.to(INTERVIEWER)[-1].subject == "Please help schedule Asha Rao"


async def test_decline_and_invalid_slot(client, shortlisted):
    h = Harness(ReplyIntent(intent="decline", summary="Took another offer"))
    interview_id = await start(client, shortlisted)
    await h.worker.run_until_idle()
    await reply(client, h, "I accepted another offer, sorry.")
    assert (await interview(interview_id)).status.value == "withdrawn"


async def test_slot_number_out_of_range_goes_to_a_human(client, shortlisted):
    h = Harness(ReplyIntent(intent="accept", slot_number=7, summary="Wants slot 7"))
    interview_id = await start(client, shortlisted)
    await h.worker.run_until_idle()
    await reply(client, h, "Number 7 please")
    assert (await interview(interview_id)).status.value == "needs_human"


async def test_slot_taken_meanwhile_offers_new_times(client, shortlisted):
    h = Harness(
        ReplyIntent(intent="accept", slot_number=1, summary="Option 1"),
        ReplyIntent(intent="accept", slot_number=1, summary="Option 1 again"),
        kit(),
    )
    interview_id = await start(client, shortlisted)
    await h.worker.run_until_idle()
    first_slot = datetime(2026, 11, 24, 10, 0, tzinfo=IST)
    await DbCalendar(SessionLocal).book(  # another meeting grabs that time
        INTERVIEWER, first_slot, first_slot + timedelta(hours=1), reason="Team sync"
    )

    await reply(client, h, "Option 1 please")
    offers = [e for e in h.to("asha@example.com") if "pick a time" in e.subject]
    assert len(offers) == 2  # booking failed safely, new options sent

    await reply(client, h, "Option 1 of the new ones")
    iv = await interview(interview_id)
    assert iv.status.value == "booked" and iv.starts_at != first_slot


async def test_duplicate_and_late_webhooks_are_harmless(client, shortlisted):
    h = Harness(ReplyIntent(intent="accept", slot_number=1, summary="Option 1"), kit())
    await start(client, shortlisted)
    await h.worker.run_until_idle()

    first = await reply(client, h, "1 works", message_id="<m1@mail>")
    again = await reply(client, h, "1 works", message_id="<m1@mail>")  # provider retried
    late = await reply(client, h, "Actually, 2?", message_id="<m2@mail>")  # after booking

    assert first == {"queued": True} and again == {"queued": False} and late == {"queued": True}
    confirmations = [e for e in h.outbox.sent if e.subject.startswith("Confirmed")]
    assert len(confirmations) == 1  # booked once; the late reply was ignored safely


async def test_starting_twice_does_not_email_twice(client, shortlisted):
    h = Harness()
    interview_id = await start(client, shortlisted)
    res = await client.post(
        f"/api/jobs/{shortlisted['job_id']}/interviews", json={"interviewer_email": INTERVIEWER}
    )
    assert res.json()["started"] == 0  # the candidate already has an interview
    await h.worker.run_until_idle()

    await h.handlers["start_scheduling"]({"interview_id": interview_id})  # a retried task
    assert len(h.to("asha@example.com")) == 1  # already waiting for a reply: no second email


async def test_no_free_slots_escalates(client, shortlisted):
    h = Harness()
    await DbCalendar(SessionLocal).book(
        INTERVIEWER, NOW, NOW + timedelta(days=10), reason="Holiday"
    )
    interview_id = await start(client, shortlisted)
    await h.worker.run_until_idle()
    assert (await interview(interview_id)).status.value == "needs_human"
    assert h.to("asha@example.com") == []


async def test_inbound_webhook_security(client, shortlisted):
    await start(client, shortlisted)
    body = {"from_email": "x@y.z", "to": "interview+nope@hireflow.local", "text": "hi"}
    bad = await client.post("/api/inbound-email", json=body, headers={"X-Inbound-Secret": "wrong"})
    unknown = await client.post("/api/inbound-email", json=body, headers=SECRET)
    assert bad.status_code == 401 and unknown.status_code == 404


async def test_assistant_hands_work_to_the_scheduler(app, client, shortlisted):
    script = [
        tool_call(
            "schedule_interviews",
            {"job_id": shortlisted["job_id"], "interviewer_email": INTERVIEWER},
        ),
        tool_call("interview_status", {"job_id": shortlisted["job_id"]}, "call_2"),
        AIMessage("Scheduling started for 1 candidate (Asha Rao)."),
    ]
    app.dependency_overrides[get_chat_model] = lambda: FakeToolModel(messages=iter(script))

    res = await client.post("/api/assistant/chat", json={"message": "Schedule interviews, job 1"})

    assert res.json()["tools_used"] == ["schedule_interviews", "interview_status"]
    rows = (await client.get(f"/api/jobs/{shortlisted['job_id']}/interviews")).json()
    assert [(r["candidate"], r["status"]) for r in rows] == [("Asha Rao", "scheduling")]