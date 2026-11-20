"""Index the images, check for duplicates and freeze a stratified split by ride.

All windows of one ride go to the same split (train, validation or test).
Windows of the same ride look alike: splitting them would leak test information
into training and make the scores look better than they really are.
"""

import hashlib
import json
import random
from pathlib import Path

from PIL import Image

from trajectory_mlops.config import CLASSES, read_params

EXTENSIONS = {".png", ".jpg", ".jpeg"}
MIN_RIDES_PER_CLASS = 3  # at least one ride for each of train, validation and test


def ride_id(path: Path) -> str:
    """'<RIDE_ID>_w03.png' -> '<RIDE_ID>'. A file without '_w' counts as its own ride."""
    return path.stem.rsplit("_w", 1)[0]


def split_rides(rides: list[str], train_fraction: float, val_fraction: float) -> dict[str, str]:
    """Give each ride a split; validation and test always get at least one ride."""
    n_val = max(1, round(len(rides) * val_fraction))
    n_test = max(1, round(len(rides) * (1 - train_fraction - val_fraction)))
    n_train = len(rides) - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Not enough rides ({len(rides)}) for train, validation and test.")
    return {
        ride: "train" if index < n_train else "validation" if index < n_train + n_val else "test"
        for index, ride in enumerate(rides)
    }


def prepare(params: dict) -> dict:
    cfg = params["data"]
    root = Path(cfg["raw_dir"])
    train_fraction = cfg["train_fraction"]
    val_fraction = cfg["validation_fraction"]
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("Train/validation fractions must leave a non-empty test split.")
    if not root.is_dir():
        raise ValueError(f"Dataset not found: {root}. See data/README.md.")
    unknown = {p.name for p in root.iterdir() if p.is_dir()} - set(CLASSES)
    if unknown:
        raise ValueError(f"Unexpected class folders: {sorted(unknown)}")
    records, hashes = [], set()
    rng = random.Random(cfg["seed"])
    for label in CLASSES:
        files = sorted(p for p in (root / label).rglob("*") if p.suffix.lower() in EXTENSIONS)
        rides = sorted({ride_id(p) for p in files})
        if len(rides) < MIN_RIDES_PER_CLASS:
            raise ValueError(
                f"{label}: at least {MIN_RIDES_PER_CLASS} rides needed, found {len(rides)}."
            )
        rng.shuffle(rides)
        split_of = split_rides(rides, train_fraction, val_fraction)
        for path in files:
            with Image.open(path) as image:
                image.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate image: {path}. Remove duplicates before splitting.")
            hashes.add(digest)
            records.append(
                {
                    "path": path.as_posix(),
                    "label": label,
                    "ride_id": ride_id(path),
                    "split": split_of[ride_id(path)],
                    "sha256": digest,
                }
            )
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {"classes": list(CLASSES), "fingerprint": fingerprint, "images": records}
    destination = Path(cfg["manifest"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(records)} images indexed; fingerprint={fingerprint}")
    return manifest


if __name__ == "__main__":
    prepare(read_params())
