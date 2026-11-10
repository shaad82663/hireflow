from hireflow.agents.intake import JobDraft, draft_job
from hireflow.llm.client import get_chat_model
from hireflow.models.job import EmploymentType

CANNED = JobDraft(
    title="Backend Engineer",
    location="Pune",
    remote=False,
    employment_type=EmploymentType.FULL_TIME,
    description="Join us to build APIs.",
    must_have=["Python", "SQL", "REST APIs"],
    nice_to_have=["AWS"],
    biased_phrases_found=["young and energetic"],
)


class FakeStructuredModel:
    """Pretends to be an LLM. Real model calls cost money and give different answers each time,
    so unit tests use a fake that always returns the same JobDraft."""

    def __init__(self, answer: JobDraft) -> None:
        self.answer = answer
        self.calls: list = []

    def with_structured_output(self, schema):
        assert schema is JobDraft
        return self._respond  # a plain function works as a step in a LangChain chain

    def _respond(self, prompt_value):
        self.calls.append(prompt_value.to_messages())
        return self.answer


async def test_draft_job_uses_prompt_and_schema():
    fake = FakeStructuredModel(CANNED)
    notes = "need a young and energetic python guy in pune"
    draft = await draft_job(fake, notes=notes, company="Acme")

    assert draft.title == "Backend Engineer"
    system_msg, human_msg = fake.calls[0]
    assert "Acme" in system_msg.content
    assert "young and energetic" in human_msg.content


async def test_draft_endpoint(app, client):
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredModel(CANNED)
    notes = "need a backend dev who knows SQL well"
    res = await client.post("/api/jobs/draft", json={"notes": notes})
    assert res.status_code == 200
    assert res.json()["biased_phrases_found"] == ["young and energetic"]


async def test_draft_endpoint_rejects_short_notes(app, client):
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredModel(CANNED)
    res = await client.post("/api/jobs/draft", json={"notes": "too short"})
    assert res.status_code == 422