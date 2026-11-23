from pathlib import Path

import pytest

from trajectory_mlops.config import CLASSES
from trajectory_mlops.ui_client import ApiClient, ApiError

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[2] / "app/streamlit_app.py"


def test_empty_app_shows_model_status(monkeypatch):
    monkeypatch.setattr(
        ApiClient,
        "model_info",
        lambda _: {
            "classes": list(CLASSES),
            "model_version": "4",
            "max_upload_bytes": 5242880,
            "max_image_pixels": 4000000,
        },
    )
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert "version 4" in app.success[0].value
    assert "Choose a trajectory" in app.info[0].value


def test_unavailable_api_shows_recovery_message(monkeypatch):
    def unavailable(_):
        raise ApiError("No model is ready yet.")

    monkeypatch.setattr(ApiClient, "model_info", unavailable)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert "No model" in app.warning[0].value


def test_upload_classify_renders_scores(monkeypatch, png_bytes):
    from io import BytesIO

    import streamlit as st

    from trajectory_mlops.api.schemas import Prediction

    class Upload(BytesIO):
        name = "window.png"
        type = "image/png"

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload(png_bytes))
    monkeypatch.setattr(
        ApiClient,
        "model_info",
        lambda _: {
            "classes": list(CLASSES),
            "model_version": "4",
            "max_upload_bytes": 5242880,
            "max_image_pixels": 4000000,
        },
    )
    monkeypatch.setattr(
        ApiClient,
        "predict",
        lambda *args: Prediction(
            label="Safe",
            confidence=0.6,
            model_version="4",
            probabilities=dict(zip(CLASSES, [0.6, 0.1, 0.1, 0.1, 0.1], strict=True)),
        ),
    )
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "Safe"
    assert app.metric[1].value == "60.0%"
    assert len(app.dataframe) == 1


def test_invalid_upload_disables_prediction(monkeypatch):
    from io import BytesIO

    import streamlit as st

    class Upload(BytesIO):
        name = "broken.png"
        type = "image/png"

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload(b"broken"))
    monkeypatch.setattr(
        ApiClient,
        "model_info",
        lambda _: {
            "classes": list(CLASSES),
            "model_version": "4",
            "max_upload_bytes": 5242880,
            "max_image_pixels": 4000000,
        },
    )
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert "invalid" in app.error[0].value
    assert app.button[0].disabled
