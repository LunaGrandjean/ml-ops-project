"""Intégration sur données synthétiques : aucune métrique de projet revendiquée."""

import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.ml


def test_training_registry_and_real_predictor(dataset_params, monkeypatch):
    pytest.importorskip("torch")
    mlflow = pytest.importorskip("mlflow")
    from PIL import Image

    from trajectory_mlops.api.predictor import Predictor
    from trajectory_mlops.config import Settings
    from trajectory_mlops.data.prepare import prepare
    from trajectory_mlops.models.evaluate import evaluate
    from trajectory_mlops.models.train import train

    tracking = "sqlite:///" + (Path.cwd() / "tracking.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "synthetic-test")
    Path("params.yaml").write_text(yaml.safe_dump(dataset_params), encoding="utf-8")
    prepare(dataset_params)
    train()
    evaluate()
    registry = json.loads(Path("artifacts/registry.json").read_text(encoding="utf-8"))
    settings = Settings(model_uri=f"models:/{registry['name']}/{registry['version']}")
    predictor = Predictor(settings)
    with Image.open("data/raw/Safe/0.png") as image:
        result = predictor.predict(image)
    assert len(result["probabilities"]) == 5
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
    mlflow.end_run()
