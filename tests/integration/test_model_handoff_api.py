"""Serve Hoang's actual bundled weights against an isolated local registry."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from trajectory_mlops.api.main import create_app
from trajectory_mlops.config import CLASSES, Settings

pytestmark = pytest.mark.ml
ROOT = Path(__file__).parents[2]


def test_handoff_register_load_and_http_prediction(tmp_path, monkeypatch, png_bytes):
    pytest.importorskip("torch")
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_NAME", "handoff-test")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "isolated-handoff-test")
    tracking = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    with ZipFile(ROOT / "docs/handoff/person2-model-handoff.zip") as archive:
        archive.extractall(tmp_path / "bundle")
    bundle = tmp_path / "bundle/person2-model-handoff"
    spec = importlib.util.spec_from_file_location(
        "bundle_import", ROOT / "scripts/register_model_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = module.register(bundle, tracking)
    settings = Settings(
        model_uri=f"models:/handoff-test/{version}", mlflow_tracking_uri=tracking, _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200
        info = client.get("/model").json()
        assert info["model_version"] == version
        assert info["image_size"] == 128
        response = client.post("/predict", files={"file": ("window.png", png_bytes, "image/png")})
        assert response.status_code == 200
        result = response.json()
        assert result["model_version"] == version
        assert set(result["probabilities"]) == set(CLASSES)
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    mlflow.end_run()
