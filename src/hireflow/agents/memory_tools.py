from langchain_core.tools import BaseTool, tool
from langgraph.store.base import BaseStore

from hireflow.memory.preferences import (
    PreferenceRejected,
    delete_preference,
    list_preferences,
    save_preference,
)


def build_memory_tools(store: BaseStore, recruiter_id: str) -> list[BaseTool]:
    """Tools bound to ONE recruiter. The agent cannot read or change anyone else's memories,
    because recruiter_id comes from the request, not from the LLM."""

    @tool
    async def remember_preference(preference: str) -> dict:
        """Save a lasting preference about how THIS recruiter likes to work, e.g. 'shortlists of
        5 candidates' or 'interviews only after 2pm'. Only use this when the recruiter asks you
        to remember something. Never save facts about candidates."""
        try:
            return {"saved": await save_preference(store, recruiter_id, preference)}
        except PreferenceRejected as e:
            return {"error": str(e)}

    @tool
    async def list_my_preferences() -> list[dict]:
        """List the preferences saved for this recruiter, with their ids."""
        return await list_preferences(store, recruiter_id) or [{"info": "No saved preferences."}]

    @tool
    async def forget_preference(preference_id: str) -> dict:
        """Delete one saved preference by its id (use list_my_preferences to find the id)."""
        if await delete_preference(store, recruiter_id, preference_id):
            return {"deleted": preference_id}
        return {"error": f"No preference with id {preference_id}"}

    return [remember_preference, list_my_preferences, forget_preference]