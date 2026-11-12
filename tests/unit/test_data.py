import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from trajectory_mlops.config import CLASSES
from trajectory_mlops.data.images import image_array
from trajectory_mlops.data.prepare import prepare


def test_preprocessing_contract():
    array = image_array(Image.new("L", (20, 10), 255), 16)
    assert array.shape == (3, 16, 16)
    assert array.dtype == np.float32
    assert np.all(array == 1)


def test_deterministic_disjoint_stratified_split(dataset_params):
    first = prepare(dataset_params)
    assert first == prepare(dataset_params)
    splits = {}
    for name in ("train", "validation", "test"):
        rows = [row for row in first["images"] if row["split"] == name]
        assert {row["label"] for row in rows} == set(CLASSES)
        splits[name] = {row["sha256"] for row in rows}
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["train"].isdisjoint(splits["validation"])
    assert splits["test"].isdisjoint(splits["validation"])


def test_duplicate_refused(dataset_params):
    shutil.copyfile("data/raw/Safe/0.png", "data/raw/Tandem/copy.png")
    with pytest.raises(ValueError, match="dupliquée"):
        prepare(dataset_params)


def test_missing_class_refused(dataset_params):
    shutil.rmtree(Path("data/raw/Reckless"))
    with pytest.raises(ValueError, match="Reckless"):
        prepare(dataset_params)
