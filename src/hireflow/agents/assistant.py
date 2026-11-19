from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel

SYSTEM_PROMPT = """You are HireFlow's recruiting assistant. You help recruiters manage jobs.
Rules:
- Use tools to look up facts. Never invent job ids, numbers or candidate details.
- If a tool returns an error, explain it simply and suggest what the recruiter can do.
- Only take an action that changes data (like closing a job) when the recruiter clearly asks.
- Keep answers short and concrete."""


class AssistantReply(BaseModel):
    reply: str
    tools_used: list[str]
    thread_id: str | None = None


async def ask_assistant(
    model: BaseChatModel,
    tools: list[BaseTool],
    message: str,
    *,
    thread_id: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    extra_system: str = "",
    summarize_after: int = 30,
    keep_recent: int = 10,
) -> AssistantReply:
    """Run one agent turn. With a checkpointer and a thread_id, the agent remembers earlier
    turns of the same conversation (short-term memory)."""
    agent = create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT + extra_system,
        checkpointer=checkpointer,
        middleware=[
            # When a thread gets long, older messages are replaced by one summary written by
            # the model, so every call stays small and cheap.
            SummarizationMiddleware(
                model, trigger=("messages", summarize_after), keep=("messages", keep_recent)
            )
        ],
    )
    config = {"recursion_limit": 12}  # hard stop: at most ~12 graph steps per question
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    result = await agent.ainvoke({"messages": [{"role": "user", "content": message}]}, config)

    messages = result["messages"]
    # only this turn's tool calls: everything after the newest human message
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    used = [
        c["name"] for m in messages[last_human:] if isinstance(m, AIMessage) for c in m.tool_calls
    ]
    return AssistantReply(reply=messages[-1].text, tools_used=used, thread_id=thread_id)