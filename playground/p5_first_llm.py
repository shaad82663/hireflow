"""Part 3.1 — Your first LLM calls. Needs a working .env (LLM_PROVIDER, LLM_MODEL, API key).
Run from the project root: uv run python playground/p5_first_llm.py"""

import asyncio

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage

from hireflow.llm.client import get_chat_model

SYSTEM = "You are a concise recruiting assistant. Answer in at most 3 sentences."


def ask(model: BaseChatModel, question: str) -> str:
    """1) Plain call: list of messages in, one AIMessage out."""
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    reply = model.invoke(messages)
    print("tokens used:", reply.usage_metadata)  # input/output tokens = what you pay for
    return reply.text


def stream(model: BaseChatModel, question: str) -> str:
    """2) Streaming: print tokens as they arrive (what chat UIs do)."""
    parts: list[str] = []
    for chunk in model.stream([SystemMessage(SYSTEM), HumanMessage(question)]):
        print(chunk.text, end="", flush=True)
        parts.append(chunk.text)
    print()
    return "".join(parts)


async def ask_many(model: BaseChatModel, questions: list[str]) -> list[str]:
    """3) Many questions concurrently with abatch (max_concurrency protects your rate limit)."""
    batches: list[LanguageModelInput] = [[SystemMessage(SYSTEM), HumanMessage(q)] for q in questions]
    replies = await model.abatch(batches, config={"max_concurrency": 3})
    return [r.text for r in replies]


if __name__ == "__main__":
    llm = get_chat_model()
    print(ask(llm, "What is a good first screening question for a backend engineer?"))
    stream(llm, "Explain 'notice period' to a candidate from Germany.")
    print(asyncio.run(ask_many(llm, ["Define ATS.", "Define CTC.", "Define JD."])))