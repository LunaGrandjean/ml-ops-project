from fastapi.testclient import TestClient

from trajectory_mlops.api.main import create_app
from trajectory_mlops.config import CLASSES, Settings


class FakePredictor:
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
