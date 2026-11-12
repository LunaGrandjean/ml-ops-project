from io import BytesIO

import pytest
from PIL import Image

from trajectory_mlops.config import CLASSES


@pytest.fixture
def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def dataset_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for label_index, label in enumerate(CLASSES):
        folder = tmp_path / "data" / "raw" / label
        folder.mkdir(parents=True)
        for index in range(12):
            color = (label_index * 45, index * 19, (index + label_index) * 11)
            Image.new("RGB", (16, 16), color).save(folder / f"{index}.png")
    return {
        "data": {
            "raw_dir": "data/raw",
            "manifest": "data/processed/manifest.json",
            "image_size": 16,
            "seed": 42,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
        },
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001, "seed": 42},
        "evaluation": {"min_accuracy": 0.21},
    }
