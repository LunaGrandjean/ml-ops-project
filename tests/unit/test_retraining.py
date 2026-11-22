import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trajectory_mlops import retraining
from trajectory_mlops.config import CLASSES
from trajectory_mlops.promote import PromotionResult


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "trajectory-classifier")
    registry, run, output = [tmp_path / name for name in ("registry.json", "run.json", "out.json")]
    registry.write_text(
        json.dumps({"name": "trajectory-classifier", "version": "2", "run_id": "fresh"})
    )
    run.write_text(json.dumps({"run_id": "fresh"}))
    output.write_text('{"promoted": true}')
    return registry, run, output


def fake_services(monkeypatch, *, broken=False):
    mlflow = pytest.importorskip("mlflow")
    client = Mock()
    client.get_model_version.return_value = SimpleNamespace(run_id="fresh")
    monkeypatch.setattr(mlflow, "MlflowClient", Mock(return_value=client))
    engine = Mock()
    engine.predict.return_value = {
        "label": CLASSES[0],
        "confidence": 0.6,
        "probabilities": dict(zip(CLASSES, [0.6, 0.1, 0.1, 0.1, 0.1], strict=True)),
        "model_version": "2",
    }
    if broken:
        engine.predict.side_effect = RuntimeError("Broken artifact")
    factory = Mock(return_value=engine)
    monkeypatch.setattr(retraining, "Predictor", factory)
    decision = PromotionResult("2", "1", True, True, "Promoted", "models:/trajectory-classifier/2")
    promotion = Mock(return_value=decision)
    monkeypatch.setattr(retraining, "promote", promotion)
    return client, factory, promotion, decision


def test_fresh_candidate_smoked_before_promotion(candidate, monkeypatch):
    client, factory, promotion, decision = fake_services(monkeypatch)
    result = retraining.finish_candidate(*candidate, dry_run=True)
    assert result == decision
    assert json.loads(candidate[2].read_text()) == asdict(decision)
    assert factory.call_args.args[0].model_uri == decision.model_uri
    promotion.assert_called_once_with(
        "2", dry_run=True, client=client, settings=promotion.call_args.kwargs["settings"]
    )


def test_failed_model_cannot_promote_or_leave_stale_report(candidate, monkeypatch):
    _, _, promotion, _ = fake_services(monkeypatch, broken=True)
    with pytest.raises(RuntimeError, match="Broken artifact"):
        retraining.finish_candidate(*candidate)
    promotion.assert_not_called()
    assert not candidate[2].exists()


def test_registry_run_mismatch_cannot_promote(candidate, monkeypatch):
    client, factory, promotion, _ = fake_services(monkeypatch)
    client.get_model_version.return_value.run_id = "old-run"
    with pytest.raises(ValueError, match="different run"):
        retraining.finish_candidate(*candidate)
    factory.assert_not_called()
    promotion.assert_not_called()
    assert not candidate[2].exists()


@pytest.mark.parametrize(
    "key,value", [("name", "other"), ("version", "@production"), ("run_id", "old")]
)
def test_invalid_candidate_metadata_cannot_promote(candidate, monkeypatch, key, value):
    _, factory, promotion, _ = fake_services(monkeypatch)
    metadata = json.loads(candidate[0].read_text())
    metadata[key] = value
    candidate[0].write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        retraining.finish_candidate(*candidate)
    factory.assert_not_called()
    promotion.assert_not_called()
