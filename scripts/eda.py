"""Exploratory data analysis of the GBG500 rides.

Prints the key numbers and saves the charts used in docs/EDA.md to docs/eda/.
Run from the repository root: uv run python scripts/eda.py
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from trajectory_mlops.config import CLASSES, read_params
from trajectory_mlops.data.convert import load_rides

OUTPUT = Path("docs/eda")


def save(figure: Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(OUTPUT / name, dpi=120)
    print(f"saved {OUTPUT / name}")


def main() -> None:
    params = read_params()
    cfg = params["convert"]
    signals = list(cfg["signals"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, ride_labels = load_rides(cfg["parquet"], cfg["labels"])
    data["label"] = data["ride_id"].map(ride_labels)

    # 1. Size, missing values, sampling rate
    print(f"rows: {len(data):,}  rides: {data['ride_id'].nunique()}")
    empty_rows = int(data[signals].isna().all(axis=1).sum())
    print(f"empty sensor rows: {empty_rows:,} ({empty_rows / len(data):.1%} of all rows)")
    steps = data.sort_values(["ride_id", "time_index"]).groupby("ride_id")["time_index"].diff()
    print(f"most common time step: {int(steps.mode()[0])} ms")

    # 2. Class balance: rides and 30 s windows
    rides = data.groupby("ride_id").agg(label=("label", "first"), rows=("time_index", "size"))
    window = int(cfg["window_seconds"] * 1000 / cfg["sample_ms"])
    rides["minutes"] = rides["rows"] * cfg["sample_ms"] / 60000
    rides["windows"] = rides["rows"] // window
    per_class = rides.groupby("label").agg(
        rides=("rows", "size"), windows=("windows", "sum"), median_minutes=("minutes", "median")
    )
    per_class = per_class.reindex(CLASSES)
    per_class["share_of_rides"] = per_class["rides"] / per_class["rides"].sum()
    print(per_class.round(2).to_string())

    figure = Figure(figsize=(8, 4))
    left, right = figure.subplots(1, 2)
    left.bar(CLASSES, per_class["rides"], color="#4C72B0")
    left.set_title("Rides per class")
    right.bar(CLASSES, per_class["windows"], color="#DD8452")
    right.set_title("30 s windows (images) per class")
    for axis in (left, right):
        axis.tick_params(axis="x", rotation=30)
        axis.set_yscale("log")
    save(figure, "class_distribution.png")

    # 3. Ride duration per class
    figure = Figure(figsize=(7, 4))
    axis = figure.subplots()
    axis.boxplot([rides.loc[rides["label"] == c, "minutes"] for c in CLASSES], tick_labels=CLASSES)
    axis.set_ylabel("Ride duration (minutes)")
    axis.set_title("Ride duration per class")
    axis.tick_params(axis="x", rotation=30)
    save(figure, "ride_duration.png")

    # 4. Signal distributions and the fixed image ranges
    figure = Figure(figsize=(10, 5))
    axes = figure.subplots(2, 3).ravel()
    print("share of values outside the fixed image range (clipped):")
    for axis, name in zip(axes, signals, strict=True):
        low, high = cfg["ranges"][name]
        values = data[name].dropna()
        margin = (high - low) / 2  # show a bit beyond the range, piling extremes at the edges
        axis.hist(values.clip(low - margin, high + margin), bins=80, color="#55A868")
        axis.axvline(low, color="red", linestyle="--")
        axis.axvline(high, color="red", linestyle="--")
        axis.set_title(name)
        outside = float(((values < low) | (values > high)).mean())
        print(f"  {name}: {outside:.2%}")
    figure.suptitle("Signal values (red dashed lines = fixed image range)")
    save(figure, "signal_distributions.png")

    # 5. How agitated is each ride? (std of rotation signals)
    agitation = data.groupby("ride_id")[["rx", "ry", "rz"]].std().mean(axis=1)
    rides["agitation"] = agitation
    print("median rotation std per class (deg/s):")
    print(rides.groupby("label")["agitation"].median().reindex(CLASSES).round(1).to_string())
    figure = Figure(figsize=(7, 4))
    axis = figure.subplots()
    axis.boxplot(
        [rides.loc[rides["label"] == c, "agitation"] for c in CLASSES], tick_labels=CLASSES
    )
    axis.set_ylabel("Mean std of rx, ry, rz (deg/s)")
    axis.set_title("Rotation intensity per ride")
    axis.tick_params(axis="x", rotation=30)
    save(figure, "rotation_intensity.png")

    # 6. One example image per class (from the converter output)
    raw = Path(params["data"]["raw_dir"])
    figure = Figure(figsize=(9, 7))
    axes = figure.subplots(len(CLASSES), 1)
    for axis, label in zip(axes, CLASSES, strict=True):
        example = sorted((raw / label).glob("*.png"))[0]
        axis.imshow(np.asarray(Image.open(example)), cmap="gray", vmin=0, vmax=255, aspect="auto")
        axis.set_yticks([10 + 20 * i for i in range(len(signals))], signals)
        axis.set_xticks([])
        axis.set_title(label, loc="left", fontsize=10)
    save(figure, "example_images.png")


if __name__ == "__main__":
    main()
