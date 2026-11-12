"""Indexer les images, contrôler les doublons et figer un split stratifié.

Hypothèse : une image indépendante par trajectoire. Si plusieurs images viennent
du même trajet/conducteur, remplacer ce split par un split groupé AVANT l'étude.
"""

import hashlib
import json
import random
from pathlib import Path

from PIL import Image

from trajectory_mlops.config import CLASSES, read_params

EXTENSIONS = {".png", ".jpg", ".jpeg"}


def prepare(params: dict) -> dict:
    cfg = params["data"]
    root = Path(cfg["raw_dir"])
    train_fraction = cfg["train_fraction"]
    val_fraction = cfg["validation_fraction"]
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("Les fractions train/validation doivent laisser un test non vide.")
    if not root.is_dir():
        raise ValueError(f"Dataset absent : {root}. Voir data/README.md.")
    unknown = {p.name for p in root.iterdir() if p.is_dir()} - set(CLASSES)
    if unknown:
        raise ValueError(f"Dossiers de classes inattendus : {sorted(unknown)}")
    records, hashes = [], set()
    rng = random.Random(cfg["seed"])
    for label in CLASSES:
        files = sorted(p for p in (root / label).rglob("*") if p.suffix.lower() in EXTENSIONS)
        if len(files) < 10:
            raise ValueError(f"{label} : au moins 10 images nécessaires, trouvé {len(files)}.")
        rng.shuffle(files)
        n_train = int(len(files) * train_fraction)
        n_val = int(len(files) * val_fraction)
        if min(n_train, n_val, len(files) - n_train - n_val) < 1:
            raise ValueError(f"{label} : split vide, ajuster les fractions ou ajouter des images.")
        for index, path in enumerate(files):
            with Image.open(path) as image:
                image.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Image dupliquée : {path}. Supprimer les doublons avant split.")
            hashes.add(digest)
            split = (
                "train" if index < n_train else "validation" if index < n_train + n_val else "test"
            )
            records.append(
                {"path": path.as_posix(), "label": label, "split": split, "sha256": digest}
            )
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {"classes": list(CLASSES), "fingerprint": fingerprint, "images": records}
    destination = Path(cfg["manifest"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(records)} images indexées ; fingerprint={fingerprint}")
    return manifest


if __name__ == "__main__":
    prepare(read_params())
