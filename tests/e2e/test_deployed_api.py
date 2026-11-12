import os
from pathlib import Path

import httpx
import pytest

from trajectory_mlops.config import CLASSES

pytestmark = pytest.mark.e2e


def test_deployed_prediction():
    url = os.getenv("E2E_API_URL")
    path = os.getenv("E2E_IMAGE_PATH")
    if not url or not path:
        pytest.skip("Définir E2E_API_URL et E2E_IMAGE_PATH pour tester le service réel.")
    image = Path(path)
    import mimetypes

    with httpx.Client(base_url=url, timeout=60) as client:
        client.get("/health/ready").raise_for_status()
        response = client.post(
            "/predict",
            files={"file": (image.name, image.read_bytes(), mimetypes.guess_type(image.name)[0])},
        )
        response.raise_for_status()
        result = response.json()
    assert result["label"] in CLASSES
    assert set(result["probabilities"]) == set(CLASSES)
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
