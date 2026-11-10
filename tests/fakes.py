"""Fake LLMs for tests. Real models cost money and answer differently each time."""

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeToolModel(GenericFakeChatModel):
    """Replays scripted AIMessages (which may contain tool calls).
    create_agent calls bind_tools(); the fake just ignores the schemas."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """An AIMessage in which the 'model' asks to run one tool."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


class FakeStructuredModel:
    """Returns one fixed Pydantic object from with_structured_output()."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.prompts: list = []

    def with_structured_output(self, schema):
        assert isinstance(self.answer, schema)
        return self._respond

    def _respond(self, prompt_value):
        self.prompts.append(prompt_value.to_string())
        return self.answer