import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from hireflow.agents.screening.scoring import CandidateScore, Criterion, build_rubric, score_resume
from hireflow.models.application import Application, ApplicationStage
from hireflow.models.screening import ScreeningDecision, ScreeningScore
from hireflow.services import jobs as job_service

class ScreeningState(TypedDict):  # fix: no total=False, so input keys are required and safe to read
    # input
    job_id: int
    shortlist_size: int
    min_score: float
    # filled by the graph
    rubric: NotRequired[list[Criterion]]  # fix: NotRequired = not set until a node fills it
    application_ids: NotRequired[list[int]]  # fix: NotRequired
    scores: NotRequired[Annotated[list[CandidateScore], operator.add]]  # parallel scorers append here  # fix: NotRequired
    ranking: NotRequired[list[CandidateScore]]  # fix: NotRequired
    proposal: NotRequired[list[int]]  # fix: NotRequired
    decision: NotRequired[dict]  # fix: NotRequired
    result: NotRequired[dict]  # fix: NotRequired


class ScoreTask(TypedDict):
    """The small input each parallel scorer gets. IDs, not resume text: checkpoints stay small."""

    application_id: int
    rubric: list[Criterion]

def build_screening_graph(
        model: BaseChatModel,
        session_factory: async_sessionmaker[AsyncSession],
        checkpointer: BaseCheckpointSaver | None = None
):

    async def load_job(state: ScreeningState) -> dict:
        async with session_factory() as session:
            job = await job_service.get_job(session, state["job_id"])
            stmt = (
                select(Application.id)
                .where(Application.job_id == job.id, Application.stage == ApplicationStage.APPLIED)
                .order_by(Application.id)
            )
            ids = list((await session.execute(stmt)).scalars().all())
            return {"rubric": build_rubric(job), "application_ids": ids}

    async def score_candidate(state: ScoreTask) -> dict:  # fix: param must be named "state" for add_node
        async with session_factory() as session:
            app = await session.get(
                Application, state["application_id"], options=[selectinload(Application.candidate)]
            )
        if app is None:  # fix: session.get can return None
            raise ValueError(f"Application {state['application_id']} not found")
        score = await score_resume(
            model, state["rubric"], app.id, app.resume_text, app.candidate.name
        )
        return {"scores": [score]}

    def rank(state: ScreeningState) -> dict:
        ranked = sorted(
            state.get("scores", []),
            key=lambda s: (s["must_haves_met"], s["total"]),
            reverse=True,
        )
        proposal = [
            s["application_id"]
            for s in ranked
            if s["must_haves_met"] and s["total"] >= state["min_score"]
        ][: state["shortlist_size"]]
        return {"ranking": ranked, "proposal": proposal}

    def human_review(state: ScreeningState) -> dict:
        decision = interrupt(
            {"proposal": state.get("proposal", []), "ranking": state.get("ranking", [])}  # fix: .get for NotRequired keys
        )
        return {"decision": decision}

    async def apply_decision(state: ScreeningState, config: RunnableConfig) -> dict:
        decision = state.get("decision", {})  # fix: .get for NotRequired key
        shortlist, reject = set(decision["shortlist"]), set(decision["reject"])
        thread_id = config.get("configurable", {})["thread_id"]  # fix: "configurable" is optional in RunnableConfig

        async with session_factory() as session:
            for s in state.get("ranking", []):  # fix: .get for NotRequired key
                app_id = s["application_id"]
                if app_id in shortlist:
                    outcome, stage = ScreeningDecision.SHORTLISTED, ApplicationStage.SHORTLISTED
                elif app_id in reject:
                    outcome, stage = ScreeningDecision.REJECTED, ApplicationStage.REJECTED
                else:
                    outcome, stage = ScreeningDecision.UNDECIDED, None

                if stage is not None:
                    app = await session.get(Application, app_id)
                    if app is not None:  # fix: session.get can return None
                        app.stage = stage
                session.add(
                    ScreeningScore(
                        thread_id=thread_id,
                        application_id=app_id,
                        total=s["total"],
                        must_haves_met=s["must_haves_met"],
                        details={"criteria": s["criteria"], "summary": s["summary"]},
                        decision=outcome,
                        reviewer_note=decision.get("note") or None,
                    )
                )
            await session.commit()
        return {"result": {"shortlisted": sorted(shortlist), "rejected": sorted(reject)}}


# routing 

    def fan_out(state: ScreeningState) -> list[Send] | str:
        ids, rubric = state.get("application_ids", []), state.get("rubric", [])  # fix: .get for NotRequired keys
        if not ids or not rubric:
            return "rank"
        return [Send("score_candidate", ScoreTask(application_id=i, rubric=rubric)) for i in ids]

    def after_rank(state: ScreeningState) -> str:
        return "human_review" if state.get("ranking") else END  # fix: .get for NotRequired key

    builder = StateGraph(ScreeningState)
    builder.add_node("load_job", load_job)
    builder.add_node("score_candidate", score_candidate)
    builder.add_node("rank", rank)
    builder.add_node("human_review", human_review)
    builder.add_node("apply_decision", apply_decision)

    builder.add_edge(START, "load_job")
    builder.add_conditional_edges("load_job", fan_out, ["score_candidate", "rank"])
    builder.add_edge("score_candidate", "rank")
    builder.add_conditional_edges("rank", after_rank, ["human_review", END])
    builder.add_edge("human_review", "apply_decision")
    builder.add_edge("apply_decision", END)
    return builder.compile(checkpointer=checkpointer)

