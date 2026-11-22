import pytest

from hireflow.evals.dataset import ScreeningDataset, load_screening_dataset


def test_golden_dataset_loads_and_is_balanced():
    dataset = load_screening_dataset()
    labels = [c.label for c in dataset.cases]
    assert len(labels) == 8
    assert {labels.count(x) for x in ("strong", "maybe")} == {2}  # 2 strong, 2 maybe, 4 weak


def test_dataset_rejects_unknown_job():
    with pytest.raises(ValueError, match="unknown job"):
        ScreeningDataset.model_validate(
            {"jobs": {}, "cases": [{"id": "x", "job": "nope", "label": "weak", "why": "-",
                                    "name": "A B", "resume": "r" * 60}]}
        )  # fmt: skip