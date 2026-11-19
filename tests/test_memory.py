import pytest
from fakes import RecordingChatModel, tool_call
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from hireflow.agents.assistant import ask_assistant
from hireflow.core.config import get_settings
from hireflow.llm.client import get_chat_model
from hireflow.memory.preferences import (
    MAX_PREFERENCES,
    PreferenceRejected,
    check_preference,
    delete_preference,
    list_preferences,
    save_preference,
)


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


# ---------- 8.2 summarization ----------


async def test_long_threads_are_summarized():
    model = scripted(*[f"answer {i}" for i in range(5)])
    saver = InMemorySaver()
    for i in range(5):
        await ask_assistant(
            model, [], f"q{i}", thread_id="t1", checkpointer=saver, summarize_after=6, keep_recent=2
        )

    last_call = model.seen[-1]
    assert "SUMMARY-OF-OLD" in last_call[1].text  # old turns were replaced by a summary
    assert "q0" not in [m.text for m in last_call]  # the oldest question is gone...
    assert last_call[-1].text == "q4"  # ...but the newest is always kept
    assert len(last_call) < 1 + 2 * 5  # far fewer than all 10 messages + system prompt


# ---------- 8.3 long-term memory: preferences ----------


@pytest.mark.parametrize(
    "text",
    [
        "Email shortlists to priya@acme.com",
        "Call me on +91 98765 43210",
        "Prefer younger candidates",
        "Only shortlist men for this role",
        "Skip married applicants",
        "no",
    ],
)
def test_unsafe_or_bad_preferences_are_rejected(text):
    with pytest.raises(PreferenceRejected):
        check_preference(text)


def test_good_preference_is_cleaned():
    assert check_preference("  Shortlists of   5,\nplease ") == "Shortlists of 5, please"


async def test_preferences_are_separate_per_recruiter():
    store = InMemoryStore()
    saved = await save_preference(store, "priya", "Interviews only after 2pm")
    await save_preference(store, "ravi", "Shortlists of 3")

    assert await list_preferences(store, "priya") == [saved]
    assert not await delete_preference(store, "ravi", saved["id"])  # ravi can't delete it
    assert await delete_preference(store, "priya", saved["id"])
    assert await list_preferences(store, "priya") == []


async def test_preference_limit():
    store = InMemoryStore()
    for i in range(MAX_PREFERENCES):
        await save_preference(store, "priya", f"Rule number {i}")
    with pytest.raises(PreferenceRejected, match="Delete one first"):
        await save_preference(store, "priya", "One too many")


async def test_memories_api(client):
    created = await client.post("/api/memories", json={"text": "Shortlists of 5"})
    assert created.status_code == 201
    pref_id = created.json()["id"]

    assert (await client.get("/api/memories")).json() == [
        {"id": pref_id, "text": "Shortlists of 5"}
    ]
    assert (await client.get("/api/memories", headers={"X-Recruiter-Id": "ravi"})).json() == []

    bad = await client.post("/api/memories", json={"text": "Prefer male candidates"})
    assert bad.status_code == 422
    assert (await client.delete(f"/api/memories/{pref_id}")).status_code == 204
    assert (await client.delete(f"/api/memories/{pref_id}")).status_code == 404


async def test_agent_saves_a_preference_and_uses_it_in_a_new_chat(app, client):
    model = scripted(
        tool_call("remember_preference", {"preference": "Shortlists of 5 candidates"}),
        "Got it, I will remember that.",
        "Here are your 5 candidates.",
    )
    app.dependency_overrides[get_chat_model] = lambda: model

    first = await client.post(
        "/api/assistant/chat", json={"message": "Always give me shortlists of 5"}
    )
    assert first.json()["tools_used"] == ["remember_preference"]

    # A brand-new conversation: no shared thread, but the preference is in the system prompt
    await client.post("/api/assistant/chat", json={"message": "Who should I interview?"})
    system_prompt = model.seen[-1][0].text
    assert "Shortlists of 5 candidates" in system_prompt


async def test_agent_cannot_save_an_unfair_preference(app, client):
    model = scripted(
        tool_call("remember_preference", {"preference": "Prefer candidates under 30 years of age"}),
        "I can't save that preference.",
    )
    app.dependency_overrides[get_chat_model] = lambda: model

    await client.post("/api/assistant/chat", json={"message": "Remember: under 30 only"})

    tool_result = model.seen[-1][-1]  # the tool's answer, as the model saw it
    assert "must not depend on these" in tool_result.text
    assert (await client.get("/api/memories")).json() == []


async def test_preferences_survive_in_postgres():
    """Same functions, real Postgres store: memories outlive the process."""
    url = get_settings().checkpoint_db_url
    async with AsyncPostgresStore.from_conn_string(url) as store:
        await store.setup()
        saved = await save_preference(store, "pg-test", "Interviews only after 2pm")
    async with AsyncPostgresStore.from_conn_string(url) as fresh_store:  # "after a restart"
        assert saved in await list_preferences(fresh_store, "pg-test")
        assert await delete_preference(fresh_store, "pg-test", saved["id"])