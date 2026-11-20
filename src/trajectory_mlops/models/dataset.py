"""Image dataset for training and evaluation, with optional training-only augmentation."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from trajectory_mlops.config import CLASSES
from trajectory_mlops.data.images import image_array


def augment(image: torch.Tensor, max_shift: float, noise_std: float) -> torch.Tensor:
    """Random small changes: shift the window in time and add a little sensor noise.

    The image width is time, so a horizontal roll = the same ride seen a few seconds
    earlier or later. Noise imitates small sensor errors. Both keep the class unchanged.
    """
    limit = int(image.shape[-1] * max_shift)
    if limit:
        shift = int(torch.randint(-limit, limit + 1, (1,)))
        image = torch.roll(image, shifts=shift, dims=-1)
    if noise_std:
        image = (image + torch.randn_like(image) * noise_std).clamp(0, 1)
    return image


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        image_size: int,
        max_shift: float = 0.0,
        noise_std: float = 0.0,
    ):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest["classes"] != list(CLASSES):
            raise ValueError("The manifest uses an incompatible class order.")
        self.records = [row for row in manifest["images"] if row["split"] == split]
        if not self.records:
            raise ValueError(f"Empty split: {split}")
        self.image_size = image_size
        self.max_shift = max_shift  # 0 = no augmentation (validation and test)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            tensor = torch.from_numpy(image_array(image, self.image_size))
        if self.max_shift or self.noise_std:
            tensor = augment(tensor, self.max_shift, self.noise_std)
        return tensor, CLASSES.index(row["label"])
