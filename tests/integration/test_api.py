from fastapi.testclient import TestClient

from trajectory_mlops.api.main import create_app
from trajectory_mlops.config import CLASSES, Settings


class FakePredictor:
    version = "test-version"

    def predict(self, image):
        assert image.mode == "RGB"
        return {
            "label": CLASSES[0],
            "confidence": 0.6,
            "probabilities": dict(zip(CLASSES, [0.6, 0.1, 0.1, 0.1, 0.1], strict=True)),
            "model_version": "test-version",
        }


def test_api_without_model_is_live_but_not_ready():
    with TestClient(create_app(Settings(model_uri="", _env_file=None))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.get("/docs").status_code == 200
        assert (
            client.post("/predict", files={"file": ("a.png", b"x", "image/png")}).status_code == 503
        )


def test_prediction_contract(png_bytes):
    with TestClient(create_app(predictor=FakePredictor())) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post("/predict", files={"file": ("a.png", png_bytes, "image/png")})
    assert response.status_code == 200
    assert response.json()["label"] == "Safe"
    assert response.json()["model_version"] == "test-version"
    assert abs(sum(response.json()["probabilities"].values()) - 1) < 1e-6


def test_invalid_and_unsupported_images(png_bytes):
    with TestClient(create_app(predictor=FakePredictor())) as client:
        assert (
            client.post("/predict", files={"file": ("a.txt", b"x", "text/plain")}).status_code
            == 415
        )
        assert (
            client.post("/predict", files={"file": ("a.png", b"x", "image/png")}).status_code == 422
        )
        assert client.post("/predict").status_code == 422


def test_size_limit(png_bytes):
    settings = Settings(max_upload_bytes=10, _env_file=None)
    with TestClient(create_app(settings, FakePredictor())) as client:
        response = client.post("/predict", files={"file": ("a.png", png_bytes, "image/png")})
        assert response.status_code == 413


def test_model_metadata_endpoint():
    with TestClient(create_app(predictor=FakePredictor())) as client:
        info = client.get("/model").json()
        assert info["model_version"] == "test-version"
        assert info["classes"] == list(CLASSES)
        assert info["max_upload_bytes"] > 0


def test_model_loading_failure_keeps_liveness(monkeypatch):
    from unittest.mock import Mock

    from trajectory_mlops.api import main

    monkeypatch.setattr(main, "Predictor", Mock(side_effect=RuntimeError("unreachable registry")))
    with TestClient(create_app(Settings(model_uri="models:/name/1", _env_file=None))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.get("/model").status_code == 503


def test_pixel_limit_before_inference(png_bytes):
    settings = Settings(max_image_pixels=100, _env_file=None)
    with TestClient(create_app(settings, FakePredictor())) as client:
        result = client.post("/predict", files={"file": ("a.png", png_bytes, "image/png")})
        assert result.status_code == 422


def test_spoofed_format_is_rejected():
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="GIF")
    with TestClient(create_app(predictor=FakePredictor())) as client:
        response = client.post(
            "/predict", files={"file": ("a.png", buffer.getvalue(), "image/png")}
        )
        assert response.status_code == 415


def test_inference_exception_has_controlled_error(png_bytes):
    from unittest.mock import Mock

    engine = Mock()
    engine.predict.side_effect = RuntimeError("internal secret path")
    with TestClient(create_app(predictor=engine)) as client:
        response = client.post("/predict", files={"file": ("a.png", png_bytes, "image/png")})
        assert response.status_code == 500
        assert "internal secret path" not in response.text


def test_inconsistent_model_result_has_controlled_error(png_bytes):
    class BrokenPredictor(FakePredictor):
        def predict(self, image):
            result = super().predict(image)
            result["confidence"] = 0.99
            return result

    with TestClient(create_app(predictor=BrokenPredictor())) as client:
        response = client.post("/predict", files={"file": ("a.png", png_bytes, "image/png")})
        assert response.status_code == 500


def test_jpeg_supported():
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("L", (16, 16), 128).save(buffer, format="JPEG")
    with TestClient(create_app(predictor=FakePredictor())) as client:
        response = client.post(
            "/predict", files={"file": ("a.jpg", buffer.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200
