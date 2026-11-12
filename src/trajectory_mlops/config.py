"""Contrat de classes partagé par le dataset, l'entraînement et l'API."""

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

CLASSES = ("Safe", "Safe Overall", "Bad Weather", "Tandem", "Reckless")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_uri: str = ""
    model_name: str = "trajectory-classifier"
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "trajectory-classification"
    max_upload_bytes: int = 5 * 1024 * 1024


def read_params(path: str = "params.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
