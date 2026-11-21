"""Register the evaluated Person 2 model in the target MLflow registry."""

import argparse
import json
from pathlib import Path

import mlflow
import yaml

from trajectory_mlops.config import Settings
from trajectory_mlops.promote import quality_gate


def register(bundle: Path, tracking_uri: str) -> str:
    config = yaml.safe_load((bundle / "params.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((bundle / "run.json").read_text(encoding="utf-8"))
    report = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
    if report["run_id"] != metadata["run_id"]:
        raise ValueError("Evaluation and model metadata must refer to the same source run.")
    if not quality_gate(report["metrics"]):
        raise ValueError("The bundled evaluation does not pass the promotion gate.")

    settings = Settings()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="GBG500 tuned model handoff") as run:
        mlflow.set_tags(
            {"source_run_id": metadata["run_id"], "data_sha256": metadata["data_sha256"]}
        )
        mlflow.log_params({**config["training"], "image_size": metadata["image_size"]})
        mlflow.log_metrics(report["metrics"])
        for name in ("params.yaml", "run.json", "evaluation.json", "confusion_matrix.png"):
            mlflow.log_artifact(str(bundle / name), "handoff")
        mlflow.log_artifacts(str(bundle / "model"), "model")
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", settings.model_name)
    return version.version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Path to the extracted model bundle")
    parser.add_argument("--tracking-uri", required=True, help="Target MLflow tracking URI")
    args = parser.parse_args()
    print(f"Registered target model version: {register(args.bundle, args.tracking_uri)}")
