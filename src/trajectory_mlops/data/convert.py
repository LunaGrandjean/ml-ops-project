"""Turn raw IMU rides (parquet) into grayscale images, one per time window.

Each ride is cut into fixed windows (e.g. 30 s). Each window becomes an image with
one horizontal band per sensor signal: light = low value, dark = high value.
The same fixed value range is used for every image, so a strong shake looks dark
in every ride (a per-image scale would hide the difference between calm and
reckless riding).

Output: data/raw/<CLASS>/<RIDE_ID>_w<NN>.png, read by prepare.py.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from trajectory_mlops.config import CLASSES, read_params

# The labels CSV uses different capitalisation than the class names in config.py.
LABEL_NAMES = {label.lower(): label for label in CLASSES}


def load_rides(parquet_path: str, labels_path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read the sensor data and return it with a {ride_id: class} mapping."""
    data = pd.read_parquet(parquet_path)
    labels = pd.read_csv(labels_path)
    unknown = set(labels["LABEL"].str.lower()) - set(LABEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown labels in {labels_path}: {sorted(unknown)}")
    ride_labels = {
        ride: LABEL_NAMES[label.lower()]
        for ride, label in zip(labels["RIDE_ID"], labels["LABEL"], strict=True)
    }
    return data, ride_labels


def clean_ride(ride: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Sort by time, drop duplicate timestamps and fill missing sensor rows."""
    ride = ride.sort_values("time_index").drop_duplicates("time_index")
    ride[signals] = ride[signals].interpolate(limit_direction="both")
    return ride


def split_windows(values: np.ndarray, window: int) -> list[np.ndarray]:
    """Cut a (time, signals) array into full windows; the incomplete tail is dropped."""
    return [values[start : start + window] for start in range(0, len(values) - window + 1, window)]


def window_to_image(window: np.ndarray, ranges: list[list[float]], band_height: int) -> Image.Image:
    """Draw one window as a grayscale image: one band per signal, one column per time step."""
    low = np.array([r[0] for r in ranges])
    high = np.array([r[1] for r in ranges])
    scaled = np.clip((window - low) / (high - low), 0, 1)  # 0 = low value, 1 = high value
    pixels = (255 * (1 - scaled)).astype(np.uint8)  # invert: high value = dark pixel
    bands = np.repeat(pixels.T, band_height, axis=0)  # (signals * band_height, time)
    return Image.fromarray(bands, mode="L")


def convert(params: dict) -> dict[str, int]:
    cfg = params["convert"]
    signals = list(cfg["signals"])
    ranges = [cfg["ranges"][name] for name in signals]
    window = int(cfg["window_seconds"] * 1000 / cfg["sample_ms"])
    output = Path(params["data"]["raw_dir"])

    data, ride_labels = load_rides(cfg["parquet"], cfg["labels"])
    # Start from an empty folder so old images never mix with new ones.
    if output.exists():
        shutil.rmtree(output)
    for label in CLASSES:
        (output / label).mkdir(parents=True)

    counts = dict.fromkeys(CLASSES, 0)
    for ride_id, ride in data.groupby("ride_id"):
        if ride_id not in ride_labels:
            raise ValueError(f"Ride {ride_id} has no label.")
        label = ride_labels[ride_id]
        values = clean_ride(ride, signals)[signals].to_numpy(dtype=np.float32)
        for index, piece in enumerate(split_windows(values, window)):
            image = window_to_image(piece, ranges, cfg["band_height"])
            image.save(output / label / f"{ride_id}_w{index:02d}.png")
            counts[label] += 1

    print(f"Images per class: {counts} (total {sum(counts.values())})")
    return counts


if __name__ == "__main__":
    convert(read_params())
