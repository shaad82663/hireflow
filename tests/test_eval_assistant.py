import pytest
from fakes import FakeStructuredModel, FakeToolModel, tool_call
from langchain_core.messages import AIMessage

from hireflow.evals.assistant import (
    JudgeVerdict,
    eval_agent_case,
    in_order,
    load_agent_cases,
)


@pytest.mark.parametrize(
    ("expected", "actual", "ok"),
    [
        (["get_job_details"], ["list_jobs", "get_job_details"], True),
        (["list_jobs", "close_job"], ["list_jobs", "get_job_details", "close_job"], True),
        (["list_jobs", "close_job"], ["close_job", "list_jobs"], False),
        ([], ["list_jobs"], True),
        (["close_job"], [], False),
    ],
)
def test_in_order(expected, actual, ok):
    assert in_order(expected, actual) is ok


def test_agent_cases_load():
    cases = load_agent_cases()
    assert (
        "close_job" in next(c for c in cases if c.id == "question-is-not-a-request").forbidden_tools
    )


GOOD = JudgeVerdict(correct=True, invented_facts=False, reason="Matches the reference.")
CASES = {c.id: c for c in load_agent_cases()}


async def test_agent_case_passes():
    model = FakeToolModel(
        messages=iter([tool_call("close_job", {"job_id": 7}), AIMessage("Done: job 7 is closed.")])
    )
    judge = FakeStructuredModel(GOOD)

    result = await eval_agent_case(model, judge, CASES["close-on-request"])

    assert result.passed and result.tools_used == ["close_job"]
    assert "Job 7 (Frontend Engineer) is now closed." in judge.prompts[0]  # judge saw reference


async def test_agent_that_acts_without_being_asked_fails():
    model = FakeToolModel(
        messages=iter([tool_call("close_job", {"job_id": 7}), AIMessage("I closed it for you.")])
    )
    result = await eval_agent_case(
        model, FakeStructuredModel(GOOD), CASES["question-is-not-a-request"]
    )

    assert not result.passed
    assert result.forbidden_used == ["close_job"]


async def test_judge_can_fail_an_answer():
    model = FakeToolModel(messages=iter([tool_call("list_jobs", {}), AIMessage("3 jobs open.")]))
    bad = JudgeVerdict(correct=False, invented_facts=True, reason="Says 3 jobs; reference says 2.")

    result = await eval_agent_case(model, FakeStructuredModel(bad), CASES["list-open-jobs"])

    assert result.trajectory_ok and not result.passed