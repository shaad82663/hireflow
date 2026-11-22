import argparse
import asyncio
import sys
from collections import Counter

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from hireflow.agents.screening.scoring import CandidateScore, build_rubric, score_resume
from hireflow.evals.dataset import Label, ScreeningCase, ScreeningDataset, load_screening_dataset

STRONG_AT = 65.0  # total score needed (with all must-haves met) to count as "strong"
WEAK_BELOW = 35.0
LABELS: list[Label] = ["strong", "maybe", "weak"]


def label_for(score: CandidateScore) -> Label:
    """Turn a score into the same 3 labels a recruiter uses, so we can compare."""
    if score["must_haves_met"] and score["total"] >= STRONG_AT:
        return "strong"
    if score["total"] < WEAK_BELOW:
        return "weak"
    return "maybe"


class CaseResult(BaseModel):
    id: str
    expected: Label
    got: Label | None  # None = the model call failed
    total: float | None
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.got == self.expected

    @property
    def critical(self) -> bool:
        """The worst mistakes: a strong candidate called weak, or a weak one called strong."""
        return {self.expected, self.got} == {"strong", "weak"}


class ScreeningReport(BaseModel):
    results: list[CaseResult]

    @property
    def accuracy(self) -> float:
        return sum(r.correct for r in self.results) / len(self.results)

    @property
    def critical_errors(self) -> list[CaseResult]:
        return [r for r in self.results if r.critical]

    @property
    def failed_calls(self) -> list[CaseResult]:
        return [r for r in self.results if r.got is None]

    def confusion(self) -> dict[Label, Counter]:
        """confusion()["strong"]["maybe"] = how many strong cases were labelled maybe."""
        table: dict[Label, Counter] = {label: Counter() for label in LABELS}
        for r in self.results:
            table[r.expected][r.got or "error"] += 1
        return table


async def eval_case(model: BaseChatModel, dataset: ScreeningDataset, case: ScreeningCase):
    rubric = build_rubric(dataset.jobs[case.job])
    try:
        score = await score_resume(model, rubric, 0, case.resume, case.name)
    except Exception as e:  # one broken call must not hide the other results
        return CaseResult(id=case.id, expected=case.label, got=None, total=None, error=str(e))
    return CaseResult(id=case.id, expected=case.label, got=label_for(score), total=score["total"])


async def run_screening_eval(
    model: BaseChatModel, dataset: ScreeningDataset, *, concurrency: int = 4
) -> ScreeningReport:
    limit = asyncio.Semaphore(concurrency)  # at most N calls at once (rate limits)

    async def one(case: ScreeningCase) -> CaseResult:
        async with limit:
            return await eval_case(model, dataset, case)

    results = await asyncio.gather(*(one(c) for c in dataset.cases))
    return ScreeningReport(results=list(results))


def format_report(report: ScreeningReport) -> str:
    lines = [f"{'case':<28} {'expected':<8} {'got':<8} {'score':>6}"]
    for r in report.results:
        mark = "ok" if r.correct else ("CRITICAL" if r.critical else "wrong")
        total = "-" if r.total is None else f"{r.total:.1f}"
        lines.append(f"{r.id:<28} {r.expected:<8} {r.got or 'error':<8} {total:>6}  {mark}")
    lines.append("")
    lines.append(
        f"accuracy: {report.accuracy:.0%}   critical errors: {len(report.critical_errors)}"
    )
    for expected, row in report.confusion().items():
        lines.append(
            f"  {expected:<7} -> " + ", ".join(f"{k}: {v}" for k, v in sorted(row.items()))
        )
    return "\n".join(lines)


def main() -> None:
    from hireflow.llm.client import get_chat_model

    parser = argparse.ArgumentParser(description="Evaluate the screening agent")
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    args = parser.parse_args()

    report = asyncio.run(run_screening_eval(get_chat_model(), load_screening_dataset()))
    print(format_report(report))
    # Exit code 1 makes CI fail: a worse prompt or model cannot be merged by accident.
    if report.accuracy < args.min_accuracy or report.critical_errors or report.failed_calls:
        sys.exit(1)


if __name__ == "__main__":
    main()