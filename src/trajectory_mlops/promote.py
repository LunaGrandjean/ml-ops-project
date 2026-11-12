"""Promotion explicite d'une version évaluée et meilleure que les baselines."""

import argparse

from trajectory_mlops.config import Settings


def quality_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics["test_accuracy"] > max(metrics["baseline_accuracy"], metrics["random_accuracy"])
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
        raise ValueError("Promotion refusée : modèle inférieur aux seuils/baselines de test.")
    client.set_registered_model_alias(settings.model_name, "production", version)
    print(f"{settings.model_name} v{version} est maintenant @production")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    promote(parser.parse_args().version)
