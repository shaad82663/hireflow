from types import SimpleNamespace

from fakes import FakeStructuredModel

from hireflow.agents.screening.scoring import (
    CriterionScore,
    Scorecard,
    anonymise,
    build_rubric,
    evidence_is_real,
    finalise,
    score_resume,
)

JOB = SimpleNamespace(must_have=["Python", "FastAPI"], nice_to_have=["AWS"])
RUBRIC = build_rubric(JOB)
RESUME = "Built Python services with FastAPI for 4 years. Some AWS Lambda work."


def card(**scores) -> Scorecard:
    """card(Python=(4, 'Built Python services'), ...) -> a Scorecard"""
    return Scorecard(
        criteria=[
            CriterionScore(criterion=k, score=v[0], evidence=v[1]) for k, v in scores.items()
        ],
        summary="ok",
    )


def test_rubric_weights_must_haves_double():
    assert [(c["name"], c["weight"]) for c in RUBRIC] == [("Python", 2), ("FastAPI", 2), ("AWS", 1)]


def test_anonymise_hides_personal_details():
    text = "Asha Rao | asha.rao@mail.com | +91 98765 43210 | linkedin.com/in/asha\nAsha led a team."
    hidden = anonymise(text, "Asha Rao")
    assert "Asha" not in hidden and "Rao" not in hidden
    assert "[EMAIL]" in hidden and "[PHONE]" in hidden and "[LINK]" in hidden
    assert "[CANDIDATE] led a team." in hidden


def test_evidence_must_really_be_in_the_resume():
    assert evidence_is_real("python   SERVICES with fastapi", RESUME)  # case/spacing ignored
    assert not evidence_is_real("10 years of Kubernetes", RESUME)
    assert not evidence_is_real("", RESUME)


def test_finalise_computes_weighted_total():
    result = finalise(
        7,
        RUBRIC,
        card(
            Python=(5, "Built Python services"), FastAPI=(5, "with FastAPI"), AWS=(5, "AWS Lambda")
        ),
        RESUME,
    )
    assert result["total"] == 100.0
    assert result["must_haves_met"] is True


def test_invented_evidence_is_not_trusted():
    result = finalise(
        7,
        RUBRIC,
        card(
            Python=(5, "Built Python services"),
            FastAPI=(5, "FastAPI expert since 2010"),
            AWS=(0, ""),
        ),
        RESUME,
    )
    fastapi = result["criteria"][1]
    assert fastapi["score"] == 1 and fastapi["verified"] is False and fastapi["evidence"] == ""
    assert result["must_haves_met"] is False  # a must-have below 3 blocks the shortlist


def test_missing_criterion_scores_zero():
    result = finalise(7, RUBRIC, card(Python=(4, "Built Python services")), RESUME)
    assert [c["score"] for c in result["criteria"]] == [4, 0, 0]


async def test_score_resume_sends_anonymised_text():
    fake = FakeStructuredModel(card(Python=(4, "Built Python services")))
    await score_resume(fake, RUBRIC, 7, "Ravi Kumar. " + RESUME, "Ravi Kumar")
    assert "Ravi" not in fake.prompts[0]
    assert "- Python (MUST HAVE)" in fake.prompts[0]