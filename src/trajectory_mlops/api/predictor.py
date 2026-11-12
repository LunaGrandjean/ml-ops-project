"""Chargement unique du modèle ; version d'alias résolue au démarrage."""

import numpy as np
from PIL import Image

from trajectory_mlops.config import CLASSES, Settings
from trajectory_mlops.data.images import image_array


class Predictor:
    def __init__(self, settings: Settings):
        import mlflow
        import mlflow.pytorch
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        uri = settings.model_uri
        self.version = uri
        if uri.startswith("models:/") and "@" in uri:
            name, alias = uri.removeprefix("models:/").split("@", 1)
            version = MlflowClient().get_model_version_by_alias(name, alias)
            uri = f"models:/{name}/{version.version}"
            self.version = str(version.version)
        self.model = mlflow.pytorch.load_model(uri, map_location="cpu")
        self.model.eval()
        self.image_size = int(self.model.image_size)

    def predict(self, image: Image.Image) -> dict:
        import torch

        tensor = torch.from_numpy(image_array(image, self.image_size)).unsqueeze(0)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0].numpy()
        if len(probabilities) != len(CLASSES) or not np.isfinite(probabilities).all():
            raise ValueError("Sortie du modèle incompatible avec le contrat de classes.")
        index = int(probabilities.argmax())
        return {
            "label": CLASSES[index],
            "confidence": float(probabilities[index]),
            "probabilities": dict(zip(CLASSES, probabilities.tolist(), strict=True)),
            "model_version": self.version,
        }
