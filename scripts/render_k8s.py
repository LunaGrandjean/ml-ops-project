"""Rendre les manifests avec des images immuables, sans bibliothèque externe."""

import argparse
import json
import re
from pathlib import Path


def render(registry: str, tag: str, model_uri: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", registry):
        raise ValueError("registry doit être votre identifiant Docker Hub.")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError("Tag d'image invalide.")
    if not model_uri.startswith("models:/"):
        raise ValueError("Utiliser une URI MLflow : models:/nom/version ou @alias.")
    text = Path("k8s/apps.template.yaml").read_text(encoding="utf-8")
    return (
        text.replace("__REGISTRY__", registry)
        .replace("__TAG__", tag)
        .replace("__MODEL_URI_JSON__", json.dumps(model_uri))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-uri", required=True)
    args = parser.parse_args()
    output = Path("k8s/rendered/apps.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(args.registry, args.tag, args.model_uri), encoding="utf-8")
    print(output)
