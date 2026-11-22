"""Validate this run's candidate, smoke-test it, then decide promotion."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from trajectory_mlops.api.predictor import Predictor
from trajectory_mlops.config import Settings
from trajectory_mlops.promote import PromotionResult, promote


def finish_candidate(
    registry_path: Path, run_path: Path, output_path: Path, *, dry_run: bool = False
) -> PromotionResult:
    output_path.unlink(missing_ok=True)
    settings = Settings()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    version = str(registry["version"])
    if not version.isdecimal() or int(version) < 1:
        raise ValueError("Candidate must have a numbered registry version.")
    if registry["name"] != settings.model_name or registry["run_id"] != run["run_id"]:
        raise ValueError("Candidate registration does not match this training run.")
    from mlflow import MlflowClient

    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri, registry_uri=settings.mlflow_tracking_uri
    )
    candidate = client.get_model_version(settings.model_name, version)
    if candidate.run_id != run["run_id"]:
        raise ValueError("Registry version belongs to a different run.")
    engine = Predictor(
        settings.model_copy(update={"model_uri": f"models:/{settings.model_name}/{version}"})
    )
    from trajectory_mlops.api.main import Prediction

    # Verify artifact transfer and serving contract before moving the alias.
    Prediction.model_validate(engine.predict(Image.new("RGB", (128, 128), "white")))
    decision = promote(version, dry_run=dry_run, client=client, settings=settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(decision), indent=2) + "\n", encoding="utf-8")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    decision = finish_candidate(
        Path("artifacts/registry.json"),
        Path("artifacts/run.json"),
        Path("artifacts/promotion.json"),
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(decision), indent=2))
    if path := os.getenv("GITHUB_OUTPUT"):
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(f"promoted={str(decision.promoted).lower()}\n")
            stream.write(f"model_uri={decision.model_uri}\n")


if __name__ == "__main__":
    main()
