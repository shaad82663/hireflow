from collections.abc import Callable
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hireflow.agents.interview_kit import create_interview_kit
from hireflow.agents.scheduling import build_scheduling_graph
from hireflow.integrations.email import EmailSender
from hireflow.models.interview import Interview
from hireflow.scheduling.calendar import Calendar, DbCalendar
from hireflow.worker import Handler


def build_handlers(
    *,
    model: BaseChatModel,
    session_factory: async_sessionmaker[AsyncSession],
    email: EmailSender,
    checkpointer: BaseCheckpointSaver,
    calendar: Calendar | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Handler]:
    """Task kind -> async function. The worker looks up handlers here by task.kind."""
    graph = build_scheduling_graph(
        model, session_factory, calendar or DbCalendar(session_factory), email, checkpointer, clock
    )

    async def config_for(interview_id: int) -> dict:
        async with session_factory() as session:
            interview = await session.get(Interview, interview_id)
        return {"configurable": {"thread_id": interview.thread_id}}

    async def start_scheduling(payload: dict) -> None:
        config = await config_for(payload["interview_id"])
        snapshot = await graph.aget_state(config)
        if not snapshot.values:  # brand new run
            await graph.ainvoke({"interview_id": payload["interview_id"]}, config)
        elif snapshot.next and snapshot.next != ("wait_for_reply",):
            # A previous attempt crashed half-way: continue from the last saved checkpoint
            # instead of starting over (durable execution).
            await graph.ainvoke(None, config)
        # otherwise it is already waiting for a reply or finished: nothing to do

    async def resume_scheduling(payload: dict) -> None:
        config = await config_for(payload["interview_id"])
        snapshot = await graph.aget_state(config)
        if snapshot.next != ("wait_for_reply",):
            return  # not waiting (already booked, or an old/duplicate reply): ignore safely
        await graph.ainvoke(Command(resume=payload["reply"]), config)

    async def build_interview_kit(payload: dict) -> None:
        await create_interview_kit(model, session_factory, payload["interview_id"])

    return {
        "start_scheduling": start_scheduling,
        "resume_scheduling": resume_scheduling,
        "build_interview_kit": build_interview_kit,
    }