"""Exercise the release smoke command over a real local HTTP connection."""

import threading
import time
from contextlib import contextmanager
from socket import socket

import pytest
import uvicorn

from trajectory_mlops.api.main import create_app
from trajectory_mlops.api.smoke import smoke
from trajectory_mlops.config import CLASSES, Settings


class Predictor:
    version = "7"

    def predict(self, image):
        assert image.mode == "RGB"
        return {
            "label": "Safe",
            "confidence": 0.6,
            "model_version": self.version,
            "probabilities": dict(zip(CLASSES, [0.6, 0.1, 0.1, 0.1, 0.1], strict=True)),
        }


@contextmanager
def serving(app):
    with socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 30
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started, "Local API failed to start."
            yield url
        finally:
            server.should_exit = True
            thread.join(timeout=30)
            assert not thread.is_alive(), "Local API failed to stop."


def test_smoke_posts_image_and_checks_version():
    settings = Settings(model_uri="", _env_file=None)
    with serving(create_app(settings, Predictor())) as url:
        result = smoke(url, "models:/trajectory-classifier/7")
        assert result["model_version"] == "7"
        with pytest.raises(ValueError, match="different model"):
            smoke(url, "models:/trajectory-classifier/8")


def test_smoke_fails_if_model_is_unavailable():
    from urllib.error import HTTPError

    with serving(create_app(Settings(model_uri="", _env_file=None))) as url:
        with pytest.raises(HTTPError) as error:
            smoke(url)
        assert error.value.code == 503
