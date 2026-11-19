from fakes import RecordingChatModel
from langchain_core.messages import AIMessage

from hireflow.llm.client import get_chat_model


def scripted(*replies: AIMessage | str) -> RecordingChatModel:
    msgs = [AIMessage(r) if isinstance(r, str) else r for r in replies]
    return RecordingChatModel(messages=iter(msgs), seen=[])


# ---------- 8.1 short-term memory: threads ----------


async def test_same_thread_remembers_earlier_turns(app, client):
    model = scripted("Noted: Pune.", "You said Pune.")
    app.dependency_overrides[get_chat_model] = lambda: model

    first = (await client.post("/api/assistant/chat", json={"message": "I hire in Pune"})).json()
    second = await client.post(
        "/api/assistant/chat",
        json={"message": "Where do I hire?", "thread_id": first["thread_id"]},
    )

    assert second.json()["reply"] == "You said Pune."
    texts = [m.text for m in model.seen[-1]]  # what the model saw on the 2nd call
    assert "I hire in Pune" in texts and "Noted: Pune." in texts


async def test_new_thread_starts_empty(app, client):
    model = scripted("one", "two")
    app.dependency_overrides[get_chat_model] = lambda: model

    await client.post("/api/assistant/chat", json={"message": "I hire in Pune"})
    await client.post("/api/assistant/chat", json={"message": "Where do I hire?"})  # no thread_id

    assert [m.type for m in model.seen[-1]] == ["system", "human"]


async def test_thread_history_endpoint(app, client):
    app.dependency_overrides[get_chat_model] = lambda: scripted("Hello!")
    reply = (await client.post("/api/assistant/chat", json={"message": "Hi there"})).json()

    res = await client.get(f"/api/assistant/threads/{reply['thread_id']}")

    assert res.json() == [{"role": "human", "text": "Hi there"}, {"role": "ai", "text": "Hello!"}]
    missing = await client.get("/api/assistant/threads/chat-demo-nope")
    assert missing.status_code == 404


async def test_cannot_use_someone_elses_thread(app, client):
    app.dependency_overrides[get_chat_model] = lambda: scripted("hi")
    reply = (await client.post("/api/assistant/chat", json={"message": "Hi there"})).json()

    other = {"X-Recruiter-Id": "ravi"}
    res = await client.get(f"/api/assistant/threads/{reply['thread_id']}", headers=other)
    assert res.status_code == 403
    res = await client.post(
        "/api/assistant/chat",
        json={"message": "Hi there", "thread_id": reply["thread_id"]},
        headers=other,
    )
    assert res.status_code == 403


async def test_bad_recruiter_header_is_rejected(client):
    res = await client.get(
        "/api/assistant/threads/chat-x-1", headers={"X-Recruiter-Id": "Robert'); DROP"}
    )
    assert res.status_code == 422