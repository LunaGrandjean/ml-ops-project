"""Load once on CPU and pin registry aliases to a numbered model version."""

import re

import numpy as np
from PIL import Image

from trajectory_mlops.api.schemas import Prediction
from trajectory_mlops.config import CLASSES, Settings
from trajectory_mlops.data.images import image_array


class Predictor:
    def __init__(self, settings: Settings):
        import mlflow
        import mlflow.pytorch
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_registry_uri(settings.mlflow_tracking_uri)
        match = re.fullmatch(r"models:/([\w.-]+)(?:@([^/\s]+)|/([1-9][0-9]*))", settings.model_uri)
        if not match:
            raise ValueError("MODEL_URI must identify a numbered model or a registry alias.")
        name, alias, version = match.groups()
        if alias:
            client = MlflowClient(
                tracking_uri=settings.mlflow_tracking_uri, registry_uri=settings.mlflow_tracking_uri
            )
            version = str(client.get_model_version_by_alias(name, alias).version)
        self.version = version
        self.model_uri = f"models:/{name}/{version}"
        info = mlflow.models.get_model_info(self.model_uri)
        metadata = info.metadata or {}
        if tuple(metadata.get("classes", ())) != CLASSES:
            raise ValueError("Model metadata must declare the shared class order.")
        self.model = mlflow.pytorch.load_model(self.model_uri, map_location="cpu")
        self.model.eval()
        self.image_size = int(self.model.image_size)
        if not 8 <= self.image_size <= 1024:
            raise ValueError("Model image_size is outside supported bounds.")
        if metadata.get("image_size") != self.image_size:
            raise ValueError("Model metadata and preprocessing image_size disagree.")

    def predict(self, image: Image.Image) -> dict:
        import torch

        tensor = torch.from_numpy(image_array(image, self.image_size)).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(tensor)
            if tuple(logits.shape) != (1, len(CLASSES)) or not torch.isfinite(logits).all():
                raise ValueError("Model must return five finite logits for one image.")
            probabilities = logits.softmax(dim=1)[0].cpu().numpy()
        index = int(np.argmax(probabilities))
        return Prediction(
            label=CLASSES[index],
            confidence=float(probabilities[index]),
            probabilities=dict(zip(CLASSES, probabilities.tolist(), strict=True)),
            model_version=self.version,
        ).model_dump()
