from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trajectory_mlops.api.predictor import Predictor
from trajectory_mlops.config import CLASSES, Settings

pytestmark = pytest.mark.ml


@pytest.fixture
def registry(monkeypatch):
    pytest.importorskip("mlflow")
    pytest.importorskip("torch")
    import mlflow.pytorch

    model = Mock(image_size=128)
    loader = Mock(return_value=model)
    monkeypatch.setattr(mlflow.pytorch, "load_model", loader)
    info = SimpleNamespace(metadata={"classes": list(CLASSES), "image_size": 128})
    monkeypatch.setattr(mlflow.models, "get_model_info", Mock(return_value=info))
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
    monkeypatch.setattr(mlflow, "MlflowClient", Mock(return_value=client))
    return client, loader, info, model


@pytest.mark.parametrize(
    "uri", ["models:/trajectory-classifier@production", "models:/trajectory-classifier/7"]
)
def test_alias_and_number_pin_exact_artifact(registry, uri):
    _, loader, _, _ = registry
    engine = Predictor(Settings(model_uri=uri, _env_file=None))
    assert engine.version == "7"
    loader.assert_called_once_with("models:/trajectory-classifier/7", map_location="cpu")


def test_wrong_class_order_blocks_loading(registry):
    _, loader, info, _ = registry
    info.metadata["classes"] = list(reversed(CLASSES))
    with pytest.raises(ValueError, match="class order"):
        Predictor(Settings(model_uri="models:/trajectory-classifier/7", _env_file=None))
    loader.assert_not_called()


def test_wrong_preprocessing_blocks_loading(registry):
    _, _, info, _ = registry
    info.metadata["image_size"] = 64
    with pytest.raises(ValueError, match="disagree"):
        Predictor(Settings(model_uri="models:/trajectory-classifier/7", _env_file=None))


@pytest.mark.parametrize("shape,nonfinite", [((1, 4), False), ((2, 5), False), ((1, 5), True)])
def test_invalid_model_outputs_are_rejected(registry, png_bytes, shape, nonfinite):
    from io import BytesIO

    import torch
    from PIL import Image

    _, _, _, model = registry
    model.return_value = torch.full(shape, float("nan") if nonfinite else 0.0)
    engine = Predictor(Settings(model_uri="models:/trajectory-classifier/7", _env_file=None))
    with Image.open(BytesIO(png_bytes)) as image, pytest.raises(ValueError, match="logits"):
        engine.predict(image)
