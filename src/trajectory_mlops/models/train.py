"""Entraîner sur train, sélectionner sur validation, tracer dans MLflow/MinIO."""

import copy
import json
import random
import subprocess
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from trajectory_mlops.config import CLASSES, Settings, read_params
from trajectory_mlops.models.dataset import TrajectoryDataset
from trajectory_mlops.models.network import TrajectoryCNN


def git_info(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def predictions(model, loader):
    labels, predicted = [], []
    model.eval()
    with torch.inference_mode():
        for images, target in loader:
            predicted.extend(model(images).argmax(dim=1).tolist())
            labels.extend(target.tolist())
    return labels, predicted


def train():
    params, settings = read_params(), Settings()
    cfg, data = params["training"], params["data"]
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_set = TrajectoryDataset(data["manifest"], "train", data["image_size"])
    val_set = TrajectoryDataset(data["manifest"], "validation", data["image_size"])
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"])
    model = TrajectoryCNN(data["image_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run() as run:
        manifest = json.loads(Path(data["manifest"]).read_text(encoding="utf-8"))
        mlflow.log_params({**cfg, "image_size": data["image_size"], "architecture": "small-cnn"})
        mlflow.set_tags(
            {
                "git_commit": git_info("rev-parse", "HEAD"),
                "git_dirty": str(bool(git_info("status", "--porcelain"))),
                "data_sha256": manifest["fingerprint"],
                "classes": json.dumps(CLASSES),
            }
        )
        for epoch in range(cfg["epochs"]):
            model.train()
            total_loss = 0.0
            for images, target in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(images), target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
            truth, predicted = predictions(model, val_loader)
            score = f1_score(
                truth, predicted, labels=list(range(len(CLASSES))), average="macro", zero_division=0
            )
            mlflow.log_metrics(
                {
                    "train_loss": total_loss / len(train_set),
                    "val_macro_f1": score,
                    "val_accuracy": accuracy_score(truth, predicted),
                },
                step=epoch,
            )
            if score > best_f1:
                best_f1, best_state = score, copy.deepcopy(model.state_dict())
        if best_state is None:
            raise ValueError("training.epochs doit être supérieur à zéro.")
        model.load_state_dict(best_state)
        model.eval()
        scripted = torch.jit.script(model)
        Path("artifacts").mkdir(exist_ok=True)
        torch.jit.save(scripted, "artifacts/model.pt")
        sample = next(iter(val_loader))[0][:1]
        info = mlflow.pytorch.log_model(
            scripted,
            name="model",
            serialization_format="pickle",
            pip_requirements=[
                "--extra-index-url https://download.pytorch.org/whl/cpu",
                f"torch=={torch.__version__}",
                f"numpy=={np.__version__}",
                f"mlflow=={mlflow.__version__}",
            ],
            signature=infer_signature(sample.numpy(), model(sample).detach().numpy()),
            input_example=sample.numpy(),
            metadata={"classes": list(CLASSES), "image_size": data["image_size"]},
        )
        mlflow.log_artifact(data["manifest"], "data")
        mlflow.log_artifact("params.yaml", "config")
        metadata = {
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "data_sha256": manifest["fingerprint"],
            "image_size": data["image_size"],
        }
        Path("artifacts/run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train()
