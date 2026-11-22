import asyncio
import json
import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, TypeAdapter

from hireflow.agents.assistant import ask_assistant
from hireflow.evals.dataset import DATA_DIR


class AgentCase(BaseModel):
    id: str
    question: str
    expected_tools: list[str]  # must be called, in this order (other calls may be in between)
    forbidden_tools: list[str]  # must never be called for this question
    reference: str  # what a correct answer says, written by a human


def load_agent_cases() -> list[AgentCase]:
    raw = json.loads((DATA_DIR / "assistant_golden.json").read_text())
    return TypeAdapter(list[AgentCase]).validate_python(raw)


# ---------- A sandbox: same tool names and arguments as the real ones, fake data ----------


def sandbox_tools() -> list[BaseTool]:
    jobs = {
        5: {"id": 5, "title": "Senior Backend Engineer", "status": "open",
            "location": "Bengaluru", "remote": True,
            "must_have": ["5+ years of Python", "FastAPI", "PostgreSQL", "REST APIs", "pytest"]},
        7: {"id": 7, "title": "Frontend Engineer", "status": "open",
            "location": "Pune", "remote": False, "must_have": ["React", "TypeScript"]},
    }  # fmt: skip

    @tool
    def list_jobs(status: str = "open") -> list[dict]:
        """List jobs with a given status: 'draft', 'open' or 'closed'.
        Returns id, title, location and remote for each job."""
        return [
            {k: j[k] for k in ("id", "title", "location", "remote")}
            for j in jobs.values()
            if j["status"] == status
        ]

    @tool
    def get_job_details(job_id: int) -> dict:
        """Get the full details of one job by its numeric id: description, requirements, status."""
        return jobs.get(job_id, {"error": f"Job {job_id} not found"})

    @tool
    def close_job(job_id: int) -> dict:
        """Close an OPEN job so it no longer accepts applications.
        Only use this when the recruiter clearly asked to close that specific job."""
        job = jobs.get(job_id)
        if job is None:
            return {"error": f"Job {job_id} not found"}
        job["status"] = "closed"
        return {"id": job_id, "status": "closed"}

    return [list_jobs, get_job_details, close_job]


# ---------- Graders ----------


def in_order(expected: list[str], actual: list[str]) -> bool:
    """True if `expected` appears inside `actual` in the same order (gaps allowed).
    ["get_job_details"] matches ["list_jobs", "get_job_details"]."""
    remaining = iter(actual)
    return all(name in remaining for name in expected)  # `in` moves the iterator forward


class JudgeVerdict(BaseModel):
    correct: bool = Field(description="The answer agrees with every fact in the reference")
    invented_facts: bool = Field(description="The answer states facts not in the reference")
    reason: str = Field(description="One short sentence explaining the verdict")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade answers from a recruiting assistant. Compare the ANSWER with the "
            "REFERENCE written by a human expert. Judge facts only, not style or length. "
            "Extra politeness or a suggested next step is fine; extra facts (ids, names, "
            "numbers, skills) that are not in the reference count as invented.",
        ),
        ("human", "QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}"),
    ]
)


async def judge(model: BaseChatModel, case: AgentCase, answer: str) -> JudgeVerdict:
    chain = JUDGE_PROMPT | model.with_structured_output(JudgeVerdict)
    return await chain.ainvoke(
        {"question": case.question, "reference": case.reference, "answer": answer}
    )


class AgentResult(BaseModel):
    id: str
    tools_used: list[str]
    answer: str
    trajectory_ok: bool
    forbidden_used: list[str]
    verdict: JudgeVerdict | None

    @property
    def passed(self) -> bool:
        good_answer = self.verdict is not None and self.verdict.correct
        no_invention = self.verdict is not None and not self.verdict.invented_facts
        return self.trajectory_ok and not self.forbidden_used and good_answer and no_invention


async def eval_agent_case(model: BaseChatModel, judge_model: BaseChatModel, case: AgentCase):
    reply = await ask_assistant(model, sandbox_tools(), case.question)  # fresh sandbox per case
    return AgentResult(
        id=case.id,
        tools_used=reply.tools_used,
        answer=reply.reply,
        trajectory_ok=in_order(case.expected_tools, reply.tools_used),
        forbidden_used=[t for t in reply.tools_used if t in case.forbidden_tools],
        verdict=await judge(judge_model, case, reply.reply),
    )


async def run_agent_eval(
    model: BaseChatModel, judge_model: BaseChatModel, cases: list[AgentCase]
) -> list[AgentResult]:
    return [await eval_agent_case(model, judge_model, c) for c in cases]


def format_agent_report(results: list[AgentResult]) -> str:
    lines = []
    for r in results:
        verdict = r.verdict.reason if r.verdict else "-"
        lines.append(f"{'PASS' if r.passed else 'FAIL'}  {r.id:<28} tools={r.tools_used}")
        if not r.passed:
            lines.append(f"      trajectory_ok={r.trajectory_ok} forbidden={r.forbidden_used}")
            lines.append(f"      judge: {verdict}")
    lines.append(f"\n{sum(r.passed for r in results)}/{len(results)} passed")
    return "\n".join(lines)


def main() -> None:
    from hireflow.llm.client import get_chat_model

    model = get_chat_model()
    results = asyncio.run(run_agent_eval(model, model, load_agent_cases()))
    print(format_agent_report(results))
    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()