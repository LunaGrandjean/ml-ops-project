"""Test HTTP d'une API réelle avec une image choisie par l'utilisateur."""

import argparse
import json
import mimetypes
from pathlib import Path

import httpx

from trajectory_mlops.config import CLASSES


def smoke(url: str, image: Path):
    with httpx.Client(base_url=url.rstrip("/"), timeout=60) as client:
        client.get("/health/ready").raise_for_status()
        response = client.post(
            "/predict",
            files={
                "file": (
                    image.name,
                    image.read_bytes(),
                    mimetypes.guess_type(image.name)[0] or "application/octet-stream",
                )
            },
        )
        response.raise_for_status()
        result = response.json()
    assert result["label"] in CLASSES
    assert set(result["probabilities"]) == set(CLASSES)
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.url, args.image)
