"""Fake LLMs for tests. Real models cost money and answer differently each time."""

import re

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


class KeywordScorer:
    """Fake scoring LLM for the screening graph. Parallel calls arrive in any order, so it
    cannot replay a script. Instead it gives 4 points to every rubric item whose name appears
    in the resume, quoting that word as evidence, and 0 otherwise."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self._score

    def _score(self, prompt_value):
        from hireflow.agents.screening.scoring import CriterionScore

        text = prompt_value.to_string()
        self.prompts.append(text)
        rubric_part, resume_part = text.split("<resume>", 1)
        names = re.findall(r"^- (.+?) \((?:MUST HAVE|nice to have)\)$", rubric_part, re.M)
        criteria = [
            CriterionScore(
                criterion=n,
                score=4 if n.lower() in resume_part.lower() else 0,
                evidence=n if n.lower() in resume_part.lower() else "",
            )
            for n in names
        ]
        return self.schema(criteria=criteria, summary="Scored by the fake model.")


class ScriptedStructuredModel:
    """Fake LLM for agents that call with_structured_output with DIFFERENT schemas.
    Give it answers in order; each call returns the next answer of the requested type."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        def respond(prompt_value):
            self.prompts.append(prompt_value.to_string())
            for i, answer in enumerate(self.answers):
                if isinstance(answer, schema):
                    return self.answers.pop(i)
            raise AssertionError(f"No scripted answer left for {schema.__name__}")

        return respond

class RecordingChatModel(GenericFakeChatModel):
    """Like FakeToolModel, but it also remembers every list of messages it was sent.
    That lets a test check what the agent 'saw': old turns, summaries, the system prompt.
    When asked to summarize (by SummarizationMiddleware) it answers without using the script."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if "summar" in messages[-1].text.lower() and "conversation" in messages[-1].text.lower():
            from langchain_core.outputs import ChatGeneration, ChatResult

            return ChatResult(generations=[ChatGeneration(message=AIMessage("SUMMARY-OF-OLD"))])
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)