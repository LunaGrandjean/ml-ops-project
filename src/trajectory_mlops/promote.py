"""Explicit promotion of an evaluated model version that beats the baselines."""

import argparse

from trajectory_mlops.config import Settings


def quality_gate(metrics: dict[str, float]) -> bool:
    """Minimum bar to reach production.

    The classes are imbalanced (83% Safe), so accuracy is misleading: always answering
    "Safe" already scores ~0.84. The gate therefore compares macro-F1 (every class counts
    the same) with the majority baseline, and only asks accuracy to beat random guessing.
    """
    return (
        metrics["test_accuracy"] > metrics["random_accuracy"]
        and metrics["test_accuracy"] >= metrics["min_accuracy"]
        and metrics["test_macro_f1"] > metrics["baseline_macro_f1"]
    )


def promote(version: str):
    from mlflow import MlflowClient

    settings = Settings()
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    model = client.get_model_version(settings.model_name, version)
    metrics = client.get_run(model.run_id).data.metrics
    if not quality_gate(metrics):
        raise ValueError("Promotion refused: the model does not beat the test baselines.")
    client.set_registered_model_alias(settings.model_name, "production", version)
    print(f"{settings.model_name} v{version} is now @production")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    promote(parser.parse_args().version)
