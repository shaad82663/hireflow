"""Part 6.4 — Pause a graph for a human, then resume it. Run: uv run python playground/p8_interrupt.py"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class OfferState(TypedDict):
    candidate: str
    salary_lpa: float
    approved: bool
    message: str


def draft_offer(state: OfferState) -> dict:
    return {"message": f"Offer for {state['candidate']}: {state['salary_lpa']} LPA"}


def human_approval(state: OfferState) -> dict:
    print("Calling interupt 1")
    # interrupt() SAVES the state and STOPS the run here. The value you pass is shown to the human.
    # When you resume with Command(resume=x), this node runs again from the top and
    # interrupt() returns x instead of stopping. So keep code before interrupt() side-effect free.
    answer = interrupt({"question": "Send this offer?", "draft": state["message"]})
    print("Calling interupt 2")
    return {"approved": answer == "yes"}


def send_offer(state: OfferState) -> dict:
    return {"message": state["message"] + (" (sent)" if state["approved"] else " (cancelled)")}


builder = StateGraph(OfferState)
builder.add_node("draft_offer", draft_offer)
builder.add_node("human_approval", human_approval)
builder.add_node("send_offer", send_offer)
builder.add_edge(START, "draft_offer")
builder.add_edge("draft_offer", "human_approval")
builder.add_edge("human_approval", "send_offer")
builder.add_edge("send_offer", END)

# A checkpointer is REQUIRED for interrupts: the paused state must be stored somewhere.
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    # thread_id names this run. Same thread_id = same saved conversation/state.
    config = {"configurable": {"thread_id": "offer-asha"}}

    first = graph.invoke({"candidate": "Asha", "salary_lpa": 24.0, "approved": False}, config)
    print("Paused with:", first["__interrupt__"][0].value)

    print("Next node:", graph.get_state(config).next)  # ('human_approval',)

    final = graph.invoke(Command(resume="no"), config)  # the human said yes
    print(final["message"])  # Offer for Asha: 24.0 LPA (sent)