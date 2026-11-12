import pytest

from trajectory_mlops.promote import quality_gate


@pytest.mark.parametrize(
    ("accuracy", "macro_f1", "accepted"),
    [
        (0.8, 0.7, True),
        (0.6, 0.7, False),
        (0.8, 0.1, False),
        (0.2, 0.1, False),
    ],
)
def test_gate_rejects_models_below_baselines(accuracy, macro_f1, accepted):
    metrics = {
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "baseline_accuracy": 0.6,
        "baseline_macro_f1": 0.15,
        "random_accuracy": 0.2,
        "min_accuracy": 0.21,
    }
    assert quality_gate(metrics) is accepted


def test_missing_metrics_cannot_promote():
    with pytest.raises(KeyError):
        quality_gate({})
