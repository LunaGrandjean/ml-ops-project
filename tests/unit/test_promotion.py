from copy import deepcopy

import pytest

from trajectory_mlops.promote import quality_gate


@pytest.fixture
def metrics():
    return {
        "test_accuracy": 0.75,
        "test_macro_f1": 0.36,
        "baseline_accuracy": 0.84,
        "baseline_macro_f1": 0.18,
        "random_accuracy": 0.2,
        "min_accuracy": 0.21,
    }


@pytest.mark.parametrize(
    ("accuracy", "macro_f1", "accepted"),
    [
        (0.8, 0.7, True),
        (0.5, 0.7, True),  # accuracy below majority is fine when macro-F1 improves
        (0.8, 0.1, False),
        (0.2, 0.7, False),
        (0.21, 0.7, True),
    ],
)
def test_gate_rejects_models_below_baselines(metrics, accuracy, macro_f1, accepted):
    metrics.update(test_accuracy=accuracy, test_macro_f1=macro_f1)
    assert quality_gate(metrics) is accepted


@pytest.mark.parametrize(
    "key",
    ["test_accuracy", "test_macro_f1", "random_accuracy", "min_accuracy", "baseline_macro_f1"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, None, True])
def test_gate_rejects_invalid_metrics(metrics, key, value):
    metrics[key] = value
    assert not quality_gate(metrics)


@pytest.mark.parametrize(
    "key",
    ["test_accuracy", "test_macro_f1", "random_accuracy", "min_accuracy", "baseline_macro_f1"],
)
def test_gate_rejects_missing_metrics(metrics, key):
    del metrics[key]
    assert not quality_gate(metrics)


def test_gate_does_not_modify_metrics(metrics):
    original = deepcopy(metrics)
    assert quality_gate(metrics)
    assert metrics == original
