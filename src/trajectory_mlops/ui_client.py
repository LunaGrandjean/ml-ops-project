"""Small HTTP client shared by the Streamlit UI and its tests."""

from urllib.parse import urlparse

import requests
from pydantic import ValidationError

from trajectory_mlops.api.schemas import Prediction
from trajectory_mlops.config import CLASSES


class ApiError(Exception):
    """An actionable message safe to show in the user interface."""


class ApiClient:
    def __init__(self, url: str):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ApiError("API_URL must be an HTTP or HTTPS address.")
        self.url = url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = requests.request(method, self.url + path, timeout=(3, 60), **kwargs)
            messages = {
                413: "The image is too large. Choose a smaller trajectory window.",
                415: "Choose a PNG or JPEG image.",
                422: "The image is invalid or its dimensions exceed the upload limit.",
                503: "No model is ready yet. Try again after the API has loaded a model.",
            }
            if response.status_code in messages:
                raise ApiError(messages[response.status_code])
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ApiError("The API returned an invalid response.")
            return payload
        except requests.Timeout as exc:
            raise ApiError("The API took too long to respond. Please try again.") from exc
        except requests.ConnectionError as exc:
            raise ApiError("Cannot connect to the API. Check that the service is running.") from exc
        except (requests.RequestException, ValueError) as exc:
            raise ApiError("The API request failed. Try again or check service logs.") from exc

    def model_info(self) -> dict:
        info = self._request("GET", "/model")
        if (
            info.get("classes") != list(CLASSES)
            or not info.get("model_version")
            or not isinstance(info.get("max_upload_bytes"), int)
            or not isinstance(info.get("max_image_pixels"), int)
        ):
            raise ApiError("The API model information is incomplete or incompatible.")
        return info

    def predict(self, name: str, content: bytes, mime: str) -> Prediction:
        payload = self._request("POST", "/predict", files={"file": (name, content, mime)})
        try:
            return Prediction.model_validate(payload)
        except ValidationError as exc:
            raise ApiError(
                "The API returned inconsistent class scores. Check the service."
            ) from exc
