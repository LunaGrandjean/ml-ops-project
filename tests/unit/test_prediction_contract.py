from copy import deepcopy

import pytest
from pydantic import ValidationError

from trajectory_mlops.api.schemas import Prediction
from trajectory_mlops.config import CLASSES


@pytest.fixture
def response():
    return {
        "label": "Safe",
        "confidence": 0.6,
        "model_version": "2",
        "probabilities": dict(zip(CLASSES, [0.6, 0.1, 0.1, 0.1, 0.1], strict=True)),
    }


@pytest.mark.parametrize(
    "change",
    [
        {"label": "unknown"},
        {"confidence": float("nan")},
        {"confidence": 0.9},
        {"model_version": ""},
        {"probabilities": {"Safe": 1.0}},
        {"label": "Tandem", "confidence": 0.1},
    ],
)
def test_invalid_prediction_is_rejected(response, change):
    response.update(change)
    with pytest.raises(ValidationError):
        Prediction.model_validate(response)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), 0.4])
def test_invalid_class_distribution_is_rejected(response, value):
    response = deepcopy(response)
    response["probabilities"]["Reckless"] = value
    with pytest.raises(ValidationError):
        Prediction.model_validate(response)
