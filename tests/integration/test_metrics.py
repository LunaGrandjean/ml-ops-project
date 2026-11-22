from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from trajectory_mlops.api.main import create_app
from trajectory_mlops.config import CLASSES, Settings


class Predictor:
    version = "42"

    def predict(self, _):
        return {
            "label": "Safe",
            "confidence": 0.6,
            "model_version": self.version,
            "probabilities": dict(zip(CLASSES, [0.6, 0.1, 0.1, 0.1, 0.1], strict=True)),
        }


def samples(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    return [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]


def value(rows, name, **labels):
    return next(s.value for s in rows if s.name == name and s.labels == labels)


def test_success_errors_latency_and_confidence(png_bytes):
    with TestClient(create_app(predictor=Predictor())) as client:
        client.post("/predict", files={"file": ("secret-ride.png", png_bytes, "image/png")})
        client.post("/predict", files={"file": ("bad.png", b"broken", "image/png")})
        rows = samples(client)
        assert (
            value(
                rows, "trajectory_api_requests_total", method="POST", route="/predict", status="200"
            )
            == 1
        )
        assert (
            value(
                rows, "trajectory_api_requests_total", method="POST", route="/predict", status="422"
            )
            == 1
        )
        assert (
            value(
                rows,
                "trajectory_api_request_duration_seconds_count",
                method="POST",
                route="/predict",
            )
            == 2
        )
        assert value(rows, "trajectory_predictions_total", **{"class": "Safe"}) == 1
        assert value(rows, "trajectory_prediction_confidence_sum", **{"class": "Safe"}) == 0.6
        assert value(rows, "trajectory_model_ready") == 1
        assert "secret-ride" not in client.get("/metrics").text


def test_unknown_paths_have_one_bounded_label():
    with TestClient(create_app(Settings(model_uri="", _env_file=None))) as client:
        for path in ("/unknown/a", "/unknown/b?ride=private"):
            assert client.get(path).status_code == 404
        client.get("/health/live")
        client.get("/health/ready")
        rows = samples(client)
        assert (
            value(
                rows, "trajectory_api_requests_total", method="GET", route="unmatched", status="404"
            )
            == 2
        )
        assert value(rows, "trajectory_model_ready") == 0
        assert not any(
            s.labels.get("route") in {"/metrics", "/health/live", "/health/ready"} for s in rows
        )


def test_registries_are_isolated_and_failed_inference_not_a_prediction(png_bytes):
    class Broken(Predictor):
        def predict(self, _):
            raise RuntimeError("failed")

    for engine in (Broken(), Predictor()):
        with TestClient(create_app(predictor=engine)) as client:
            rows = samples(client)
            assert value(rows, "trajectory_predictions_total", **{"class": "Safe"}) == 0
            if isinstance(engine, Broken):
                assert (
                    client.post(
                        "/predict", files={"file": ("a.png", png_bytes, "image/png")}
                    ).status_code
                    == 500
                )
                rows = samples(client)
                assert value(rows, "trajectory_predictions_total", **{"class": "Safe"}) == 0
                assert (
                    value(
                        rows,
                        "trajectory_api_requests_total",
                        method="POST",
                        route="/predict",
                        status="500",
                    )
                    == 1
                )
