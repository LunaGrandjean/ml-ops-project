import numpy as np
import pandas as pd
import pytest

from trajectory_mlops.config import CLASSES
from trajectory_mlops.data.convert import (
    clean_ride,
    convert,
    load_rides,
    split_windows,
    window_to_image,
)

SIGNALS = ["ax", "ay", "az", "rx", "ry", "rz"]


def make_ride(ride_id: str, rows: int) -> pd.DataFrame:
    """A fake ride: one sample every 100 ms, simple increasing values."""
    data = {"ride_id": ride_id, "time_index": np.arange(rows) * 100}
    for number, name in enumerate(SIGNALS):
        data[name] = np.linspace(0, 1, rows) + number
    return pd.DataFrame(data)


def test_split_windows_drops_incomplete_tail():
    values = np.zeros((650, 6))
    windows = split_windows(values, 300)
    assert len(windows) == 2  # 650 rows = 2 full windows of 300 + a 50-row tail (dropped)
    assert all(window.shape == (300, 6) for window in windows)


def test_window_to_image_uses_fixed_scale_and_clips():
    window = np.array([[-1.0], [0.0], [1.0], [5.0]])  # 4 time steps, 1 signal, range [-1, 1]
    image = window_to_image(window, [[-1, 1]], band_height=2)
    pixels = np.asarray(image)
    assert image.size == (4, 2)  # width = time steps, height = signals * band_height
    assert pixels[0].tolist() == [255, 127, 0, 0]  # low = white, high = dark, 5 clipped to 1


def test_clean_ride_sorts_removes_duplicates_and_fills_gaps():
    ride = make_ride("r1", 5).iloc[[3, 0, 1, 1, 2, 4]].copy()  # shuffled, row 1 twice
    ride.loc[ride["time_index"] == 200, SIGNALS] = np.nan  # one empty sensor row
    cleaned = clean_ride(ride, SIGNALS)
    assert cleaned["time_index"].tolist() == [0, 100, 200, 300, 400]
    assert not cleaned[SIGNALS].isna().any().any()


def test_load_rides_maps_label_names(tmp_path):
    make_ride("r1", 10).to_parquet(tmp_path / "rides.parquet")
    pd.DataFrame({"RIDE_ID": ["r1"], "LABEL": ["Safe overall"]}).to_csv(
        tmp_path / "labels.csv", index=False
    )
    _, labels = load_rides(str(tmp_path / "rides.parquet"), str(tmp_path / "labels.csv"))
    assert labels == {"r1": "Safe Overall"}  # CSV spelling -> class name from config.py


def test_load_rides_rejects_unknown_label(tmp_path):
    make_ride("r1", 10).to_parquet(tmp_path / "rides.parquet")
    pd.DataFrame({"RIDE_ID": ["r1"], "LABEL": ["Drunk"]}).to_csv(
        tmp_path / "labels.csv", index=False
    )
    with pytest.raises(ValueError, match="Unknown labels"):
        load_rides(str(tmp_path / "rides.parquet"), str(tmp_path / "labels.csv"))


def test_convert_writes_one_image_per_window(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rides = pd.concat([make_ride(f"ride{index}", 650) for index in range(len(CLASSES))])
    rides.to_parquet("rides.parquet")
    labels = [label.lower() for label in CLASSES]  # lower case, like the real CSV
    pd.DataFrame({"RIDE_ID": [f"ride{i}" for i in range(len(CLASSES))], "LABEL": labels}).to_csv(
        "labels.csv", index=False
    )
    stale = tmp_path / "data" / "raw" / "Safe" / "old.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    params = {
        "data": {"raw_dir": "data/raw"},
        "convert": {
            "parquet": "rides.parquet",
            "labels": "labels.csv",
            "window_seconds": 30,
            "sample_ms": 100,
            "band_height": 20,
            "signals": SIGNALS,
            "ranges": {name: [0, 6] for name in SIGNALS},
        },
    }
    counts = convert(params)
    assert counts == dict.fromkeys(CLASSES, 2)  # 650 rows -> 2 windows of 30 s per ride
    assert not stale.exists()  # old images are removed before converting
    image = tmp_path / "data" / "raw" / "Reckless" / "ride4_w01.png"
    assert image.exists()
