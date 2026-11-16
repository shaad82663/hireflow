import operator
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

class TriageState(TypedDict):
    keyword: str
    resumes: list[str]
    matches: Annotated[list[str], operator.add]
    summary: NotRequired[str]  

class CheckTask(TypedDict):  
    keyword: str
    resume: str


def check_one(state: CheckTask) -> dict:  
    """Runs once per resume, in parallel. Receives the small task dict sent by fan_out."""
    hit = state["keyword"].lower() in state["resume"].lower()
    return {"matches": [state["resume"]] if hit else []}

def summarise(state: TriageState) -> dict:
    total, found = len(state["resumes"]), len(state["matches"])
    return {"summary": f"{found} of {total} resumes mention {state['keyword']}"}

def fan_out(state: TriageState) -> list[Send] | str:
    """Send = 'run this node with this input'. One Send per resume -> parallel branches."""
    if not state["resumes"]:
        return "summarise"
    return [Send("check_one", {"resume": r, "keyword": state["keyword"]}) for r in state["resumes"]]

builder = StateGraph(TriageState)
builder.add_node("check_one", check_one)
builder.add_node("summarise", summarise)
builder.add_conditional_edges(START, fan_out, ["check_one", "summarise"])
builder.add_edge("check_one", "summarise")
builder.add_edge("summarise", END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke(
        {
            "keyword": "Python",
            "resumes": ["Python and SQL", "Java only", "python, FastAPI"],
            "matches": [],
        }
    )
    print(result["summary"])  
    print(graph.get_graph().draw_mermaid())  