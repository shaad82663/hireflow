from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from hireflow.api.deps import RecruiterDep, StoreDep
from hireflow.memory.preferences import (
    PreferenceRejected,
    delete_preference,
    list_preferences,
    save_preference,
)

router = APIRouter(prefix="/api/memories", tags=["memory"])


class NewPreference(BaseModel):
    text: str = Field(max_length=300)


@router.get("")
async def my_memories(store: StoreDep, recruiter: RecruiterDep) -> list[dict]:
    """People must be able to see what an AI remembers about how they work."""
    return await list_preferences(store, recruiter)


@router.post("", status_code=status.HTTP_201_CREATED)
async def add_memory(payload: NewPreference, store: StoreDep, recruiter: RecruiterDep) -> dict:
    try:
        return await save_preference(store, recruiter, payload.text)
    except PreferenceRejected as e:
        raise HTTPException(422, str(e)) from e


@router.delete("/{preference_id}", status_code=status.HTTP_204_NO_CONTENT)
async def forget(preference_id: str, store: StoreDep, recruiter: RecruiterDep) -> None:
    """...and delete it."""
    if not await delete_preference(store, recruiter, preference_id):
        raise HTTPException(404, "No such preference")