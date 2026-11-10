from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
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


async def ask(model: BaseChatModel, tools: list[BaseTool], message: str) -> AssistantReply:
    """Run one agent turn: the agent may call several tools before it answers."""
    agent = create_agent(model, tools=tools, system_prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"recursion_limit": 12}
    )
    messages = result["messages"]
    used = [c['name'] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    return AssistantReply(reply=messages[-1].text, tools_used=used)