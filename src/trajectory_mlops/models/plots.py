"""Draw the confusion matrix of an evaluation report as a PNG image."""

import json
from pathlib import Path

import mlflow
from matplotlib.figure import Figure

from trajectory_mlops.config import Settings


def save_confusion_matrix(report: dict, path: str) -> str:
    """Save the test confusion matrix: rows = true class, columns = predicted class."""
    matrix, classes = report["confusion_matrix"], report["classes"]
    figure = Figure(figsize=(7, 6))
    axis = figure.subplots()
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"Test confusion matrix (macro-F1 {report['metrics']['test_macro_f1']:.3f})")
    highest = max(max(row) for row in matrix)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > highest / 2 else "black"
            axis.text(j, i, value, ha="center", va="center", color=color)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path


def attach_to_run() -> None:
    """Draw the matrix from reports/evaluation.json and add it to that report's MLflow run."""
    report = json.loads(Path("reports/evaluation.json").read_text(encoding="utf-8"))
    path = save_confusion_matrix(report, "reports/confusion_matrix.png")
    mlflow.set_tracking_uri(Settings().mlflow_tracking_uri)
    with mlflow.start_run(run_id=report["run_id"]):
        mlflow.log_artifact(path, "evaluation")
    print(f"Added {path} to MLflow run {report['run_id']}")


if __name__ == "__main__":
    attach_to_run()
