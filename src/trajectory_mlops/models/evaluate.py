"""Évaluation finale sur test et enregistrement du candidat (sans promotion)."""

import json
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from trajectory_mlops.config import CLASSES, Settings, read_params
from trajectory_mlops.models.dataset import TrajectoryDataset
from trajectory_mlops.models.train import predictions


def evaluate():
    params, settings = read_params(), Settings()
    data = params["data"]
    metadata = json.loads(Path("artifacts/run.json").read_text(encoding="utf-8"))
    model = torch.jit.load("artifacts/model.pt", map_location="cpu")
    dataset = TrajectoryDataset(data["manifest"], "test", data["image_size"])
    truth, predicted = predictions(model, DataLoader(dataset, batch_size=32))
    train_set = TrajectoryDataset(data["manifest"], "train", data["image_size"])
    counts = [sum(row["label"] == label for row in train_set.records) for label in CLASSES]
    majority = counts.index(max(counts))
    labels = list(range(len(CLASSES)))
    metrics = {
        "test_accuracy": float(accuracy_score(truth, predicted)),
        "test_macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "baseline_accuracy": float(accuracy_score(truth, [majority] * len(truth))),
        "baseline_macro_f1": float(
            f1_score(
                truth, [majority] * len(truth), labels=labels, average="macro", zero_division=0
            )
        ),
        "random_accuracy": 1 / len(CLASSES),
        "min_accuracy": float(params["evaluation"]["min_accuracy"]),
    }
    report = {
        "metrics": metrics,
        "run_id": metadata["run_id"],
        "classes": list(CLASSES),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "classification_report": classification_report(
            truth, predicted, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0
        ),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_id=metadata["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("reports/evaluation.json", "evaluation")
    version = mlflow.register_model(metadata["model_uri"], settings.model_name)
    MlflowClient().set_registered_model_alias(settings.model_name, "candidate", version.version)
    Path("artifacts/registry.json").write_text(
        json.dumps(
            {"name": settings.model_name, "version": version.version, "run_id": metadata["run_id"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Candidat enregistré : {settings.model_name} v{version.version}")


if __name__ == "__main__":
    evaluate()
