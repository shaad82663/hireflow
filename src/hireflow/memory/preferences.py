import re
from uuid import uuid4

from langgraph.store.base import BaseStore

MAX_PREFERENCES = 20


# Things a "preference" must never contain: personal data about candidates...
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
# ...or rules that would treat candidates unfairly. Simple word list: a first line of defence.
_PROTECTED = re.compile(
    r"\b(age|aged|young|younger|old|older|gender|male|female|men|women|woman|man|girls?|boys?|"
    r"married|unmarried|pregnan\w*|religio\w*|caste|hindu|muslim|christian|sikh|nationality|"
    r"disabilit\w*|disabled)\b",
    re.IGNORECASE,
)

class PreferenceRejected(Exception):
    """The message explains why, in words the recruiter (and the LLM) can understand."""


def namespace(recruiter_id: str) -> tuple[str, str]:
    """Memories are grouped per recruiter: ('preferences', 'priya') never sees 'ravi'."""
    return ("preferences", recruiter_id)


def check_preference(text: str) -> str:
    text = " ".join(text.split())
    if not 3 <= len(text) <= 300:
        raise PreferenceRejected("A preference must be between 3 and 300 characters.")
    if _EMAIL.search(text) or _PHONE.search(text):
        raise PreferenceRejected(
            "Preferences are about how you like to work. Contact details are not stored in memory."
        )
    if _PROTECTED.search(text):
        raise PreferenceRejected(
            "This looks like it is about personal characteristics such as age, gender, religion "
            "or family status. Hiring decisions must not depend on these, so it was not saved."
        )
    return text


async def save_preference(store: BaseStore, recruiter_id: str, text: str) -> dict:
    text = check_preference(text)
    existing = await store.asearch(namespace(recruiter_id), limit=MAX_PREFERENCES)
    if len(existing) >= MAX_PREFERENCES:
        raise PreferenceRejected(
            f"You already have {MAX_PREFERENCES} preferences. Delete one first."
        )
    key = uuid4().hex[:8]
    await store.aput(namespace(recruiter_id), key, {"text": text})
    return {"id": key, "text": text}

async def list_preferences(store: BaseStore, recruiter_id: str) -> list[dict]:
    items = await store.asearch(namespace(recruiter_id), limit=MAX_PREFERENCES)
    items = sorted(items, key=lambda item: item.created_at)
    return [{"id": i.key, "text": i.value["text"]} for i in items]

async def delete_preference(store: BaseStore, recruiter_id: str, preference_id: str) -> bool:
    if await store.aget(namespace(recruiter_id), preference_id) is None:
        return False
    await store.adelete(namespace(recruiter_id), preference_id)
    return True


def preferences_prompt(preferences: list[dict]) -> str:
    """Text added to the system prompt, so the agent sees the preferences in every chat."""
    if not preferences:
        return ""
    lines = "\n".join(f"- {p['text']}" for p in preferences)
    return (
        "\n\nThis recruiter's saved preferences (follow them unless the current request says "
        f"otherwise; never let them override fairness rules):\n{lines}"
    )