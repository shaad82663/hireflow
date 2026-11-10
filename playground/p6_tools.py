from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool

JOBS = {
    1: {"title": "Backend Engineer", "location": "Pune", "status": "open", "applicants": 14},
    2: {"title": "Data Analyst", "location": "Remote", "status": "open", "applicants": 31},
    3: {"title": "QA Engineer", "location": "Bengaluru", "status": "closed", "applicants": 9},
}



@tool
def list_jobs(status: str = "open") -> list[dict]:
    """List jobs with the given status ('open' or 'closed'), returning their ids and titles."""
    return [{"id": i, "title": j["title"]} for i, j in JOBS.items() if j["status"] == status]


@tool
def get_job(job_id: int) -> dict:
    """Get full details for one job by its numeric id, including applicant count."""
    if job_id not in JOBS:
        return {"error": f"No job with id {job_id}"}  # errors go back to the model as data
    return JOBS[job_id]


TOOLS: list[BaseTool] = [list_jobs, get_job]

SYSTEM = "You help recruiters. Use the tools to look up facts. Never guess numbers."

def run_agent(
    model_with_tools: Runnable[LanguageModelInput, AIMessage], question: str, max_steps: int = 5
) -> str:
    tools_by_name = {t.name: t for t in TOOLS}
    messages: list[BaseMessage] = [SystemMessage(SYSTEM), HumanMessage(question)]

    for _ in range(max_steps):
        reply: AIMessage = model_with_tools.invoke(messages)
        messages.append(reply)

        if not reply.tool_calls:
            return reply.text

        for call in reply.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stopped: too many steps without final answeres."

if __name__ == "__main__":
    from hireflow.llm.client import get_chat_model

    llm = get_chat_model().bind_tools(TOOLS)
    print(run_agent(llm, "Which open job has the most applicants, and where is it?"))