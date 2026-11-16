import json
from collections.abc import AsyncIterator
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig  
from langgraph.types import Command
from pydantic import BaseModel, Field

from hireflow.agents.screening.graph import ScreeningState, build_screening_graph  
from hireflow.api.deps import CheckpointerDep, ModelDep, SessionDep
from hireflow.db.session import SessionLocal
from hireflow.services import jobs as job_service

router = APIRouter(prefix="/api", tags=["screening"])

class StartScreening(BaseModel):
    shortlist_size: int = Field(default=3, ge=1, le=20)
    min_score: float = Field(default=60, ge=0, le=100)

class ReviewDecision(BaseModel):
    shortlist: list[int] = Field(default_factory=list, description="application ids to shortlist")
    reject: list[int] = Field(default_factory=list, description="application ids to reject")
    note: str = Field(default="", max_length=1000)


class ScreeningStatus(BaseModel):
    thread_id: str
    status: Literal["awaiting_review", "completed", "no_applicants"]
    proposal: list[int] = []
    ranking: list[dict] = []
    result: dict | None = None


def run_config(thread_id: str) -> RunnableConfig:  
    return {"configurable": {"thread_id": thread_id}, "max_concurrency": 5}

async def read_status(graph, thread_id: str) -> ScreeningStatus:
    snapshot = await graph.aget_state(run_config(thread_id))  
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Screening {thread_id} not found")
    values = snapshot.values
    if snapshot.next == ("human_review",):
        state = "awaiting_review"
    elif values.get("result") is not None:
        state = "completed"
    else:
        state = "no_applicants"

    return ScreeningStatus(
        thread_id=thread_id,
        status=state,
        proposal=values.get("proposal", []),
        ranking=values.get("ranking", []),
        result=values.get("result"),
    )

async def _start(job_id: int, payload: StartScreening, session) -> tuple[str, ScreeningState]:  
    await job_service.get_job(session, job_id) 
    thread_id = f"job{job_id}-{uuid4().hex[:8]}"
    graph_input = ScreeningState(  
        job_id=job_id, shortlist_size=payload.shortlist_size, min_score=payload.min_score
    )
    return thread_id, graph_input

@router.post("/jobs/{job_id}/screenings", response_model=ScreeningStatus, status_code=201)
async def start_screening(
    job_id: int,
    payload: StartScreening,
    session: SessionDep,
    model: ModelDep,
    checkpointer: CheckpointerDep
):
    thread_id, graph_input = await _start(job_id, payload, session)
    graph = build_screening_graph(model, SessionLocal, checkpointer)
    await graph.ainvoke(graph_input, run_config(thread_id))
    return await read_status(graph, thread_id)

@router.get("/screenings/{thread_id}", response_model=ScreeningStatus)
async def get_screening(thread_id: str, model: ModelDep, checkpointer: CheckpointerDep):
    graph = build_screening_graph(model, SessionLocal, checkpointer)
    return await read_status(graph, thread_id)

@router.post("/screenings/{thread_id}/decision", response_model=ScreeningStatus)
async def decide(
    thread_id: str, decision: ReviewDecision, model: ModelDep, checkpointer: CheckpointerDep
):
    graph = build_screening_graph(model, SessionLocal, checkpointer)
    current = await read_status(graph, thread_id)
    if current.status != 'awaiting_review':
        raise HTTPException(status.HTTP_409_CONFLICT, "This screening is not waiting for review")

    scored = {s["application_id"] for s in current.ranking}
    chosen = set(decision.shortlist) | set(decision.reject)
    if not chosen <= scored:
        raise HTTPException(422, f"Unknown application ids: {sorted(chosen - scored)}")
    if set(decision.shortlist) & set(decision.reject):
        raise HTTPException(422, "An application cannot be both shortlisted and rejected")

    await graph.ainvoke(Command(resume=decision.model_dump()), run_config(thread_id))
    return await read_status(graph, thread_id)

@router.get("/screenings/{thread_id}/history")
async def history(thread_id: str, model: ModelDep, checkpointer: CheckpointerDep) -> list[dict]:
    graph = build_screening_graph(model, SessionLocal, checkpointer)
    return [
        {
            "step": snap.metadata.get("step"),
            "next": list(snap.next),
            "checkpoint_id": snap.config["configurable"]["checkpoint_id"],
        }
        async for snap in graph.aget_state_history(run_config(thread_id))
    ]

def sse(event: str, data: dict) -> str:
    """One Server-Sent Event. The browser's EventSource API understands this format."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@router.post("/jobs/{job_id}/screenings/stream")
async def start_screening_stream(
    job_id: int,
    payload: StartScreening,
    session: SessionDep,
    model: ModelDep,
    checkpointer: CheckpointerDep,
):
    """Same as start_screening, but reports progress live while candidates are scored."""
    thread_id, graph_input = await _start(job_id, payload, session)
    graph = build_screening_graph(model, SessionLocal, checkpointer)

    async def events() -> AsyncIterator[str]:
        yield sse("started", {"thread_id": thread_id})
        # stream_mode="updates": after each node finishes, we get {node_name: its update}
        async for update in graph.astream(
            graph_input, run_config(thread_id), stream_mode="updates"
        ):
            for node, data in update.items():
                if node == "load_job":
                    yield sse("loaded", {"applicants": len(data["application_ids"])})
                elif node == "score_candidate":
                    s = data["scores"][0]
                    yield sse(
                        "scored", {"application_id": s["application_id"], "total": s["total"]}
                    )
                elif node == "__interrupt__":
                    yield sse("awaiting_review", {"proposal": data[0].value["proposal"]})
        yield sse("done", {"thread_id": thread_id})

    return StreamingResponse(events(), media_type="text/event-stream")