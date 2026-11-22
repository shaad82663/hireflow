import pytest
from fakes import KeywordScorer

from hireflow.evals.dataset import load_screening_dataset
from hireflow.evals.screening import format_report, label_for, run_screening_eval


@pytest.mark.parametrize(
    ("total", "must_haves_met", "label"),
    [(80, True, "strong"), (80, False, "maybe"), (50, True, "maybe"), (20, True, "weak")],
)
def test_label_for(total, must_haves_met, label):
    assert label_for({"total": total, "must_haves_met": must_haves_met}) == label


async def test_eval_catches_a_keyword_matcher():
    """The fake scorer only matches keywords. The eval must expose exactly that weakness."""
    report = await run_screening_eval(KeywordScorer(), load_screening_dataset())

    assert report.accuracy == 7 / 8
    assert [r.id for r in report.critical_errors] == ["weak-keyword-stuffing"]
    assert report.confusion()["weak"]["strong"] == 1
    assert "CRITICAL" in format_report(report)


async def test_failed_calls_are_reported_not_hidden():
    class BrokenModel:
        def with_structured_output(self, schema):
            async def fail(_):
                raise TimeoutError("provider timed out")

            return fail

    report = await run_screening_eval(BrokenModel(), load_screening_dataset())

    assert len(report.failed_calls) == 8
    assert report.results[0].error == "provider timed out"