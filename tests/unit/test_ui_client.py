from unittest.mock import Mock

import pytest
import requests

from trajectory_mlops.config import CLASSES
from trajectory_mlops.ui_client import ApiClient, ApiError


def mock_response(monkeypatch, *, status=200, payload=None, error=None):
    response = Mock(status_code=status)
    response.json.return_value = payload
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError()
    request = Mock(return_value=response, side_effect=error)
    monkeypatch.setattr(requests, "request", request)
    return request


@pytest.mark.parametrize(
    "status,message",
    [(413, "too large"), (415, "PNG"), (422, "invalid"), (503, "No model"), (500, "failed")],
)
def test_http_errors_become_actionable_messages(monkeypatch, status, message):
    mock_response(monkeypatch, status=status)
    with pytest.raises(ApiError, match=message):
        ApiClient("http://api:8000").predict("a.png", b"image", "image/png")


@pytest.mark.parametrize(
    "error,message",
    [
        (requests.Timeout(), "too long"),
        (requests.ConnectionError(), "Cannot connect"),
    ],
)
def test_connection_errors(monkeypatch, error, message):
    mock_response(monkeypatch, error=error)
    with pytest.raises(ApiError, match=message):
        ApiClient("http://api:8000").model_info()


def test_prediction_payload_and_scores(monkeypatch):
    payload = {
        "label": "Safe",
        "confidence": 0.6,
        "model_version": "3",
        "probabilities": dict(zip(CLASSES, [0.6, 0.1, 0.1, 0.1, 0.1], strict=True)),
    }
    request = mock_response(monkeypatch, payload=payload)
    result = ApiClient("http://api:8000/").predict("a.png", b"image", "image/png")
    assert result.model_version == "3"
    assert request.call_args.args[:2] == ("POST", "http://api:8000/predict")
    assert request.call_args.kwargs["timeout"] == (3, 60)
    assert request.call_args.kwargs["files"]["file"] == ("a.png", b"image", "image/png")


def test_inconsistent_response_not_displayed(monkeypatch):
    mock_response(monkeypatch, payload={"label": "Safe", "confidence": 0.99})
    with pytest.raises(ApiError, match="inconsistent"):
        ApiClient("http://api:8000").predict("a.png", b"image", "image/png")


def test_incomplete_model_metadata_not_ready(monkeypatch):
    mock_response(monkeypatch, payload={"classes": ["other"]})
    with pytest.raises(ApiError, match="incompatible"):
        ApiClient("http://api:8000").model_info()
