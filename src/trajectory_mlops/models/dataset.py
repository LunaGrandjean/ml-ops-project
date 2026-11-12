import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from trajectory_mlops.config import CLASSES
from trajectory_mlops.data.images import image_array


class TrajectoryDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, image_size: int):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest["classes"] != list(CLASSES):
            raise ValueError("Le manifeste utilise un ordre de classes incompatible.")
        self.records = [row for row in manifest["images"] if row["split"] == split]
        if not self.records:
            raise ValueError(f"Split vide : {split}")
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            array = image_array(image, self.image_size)
        return torch.from_numpy(array), CLASSES.index(row["label"])
