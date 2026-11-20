# Données Inside

Les données réelles ne sont pas fournies dans ce squelette. Télécharger l'archive
depuis Inside selon les droits du groupe, puis l'extraire dans `data/raw/`.
Le premier responsable data doit confirmer le format réel avant l'entraînement.

Contrat attendu (noms exacts, espaces et majuscules inclus) :

```text
raw/
├── Safe/*.png
├── Safe Overall/*.png
├── Bad Weather/*.png
├── Tandem/*.png
└── Reckless/*.png
```

PNG et JPEG acceptés. Au moins 10 images par classe pour le split initial 70/15/15.
`prepare.py` vérifie les images, refuse les doublons binaires, produit un manifeste
avec SHA-256 et fige les splits de manière déterministe. Il ne supprime aucune donnée.

**À confirmer avant toute mesure** : une image correspond-elle à une trajectoire
indépendante ? Si plusieurs images partagent un trajet, conducteur ou session,
ajouter cet identifiant au manifeste et effectuer un split par groupe. Le split
actuel par image ne protège pas de cette fuite. Conserver le jeu de test final
hors des choix d'hyperparamètres et des expérimentations répétées.

Git ignore les images. DVC peut les versionner via un stockage partagé autorisé.

## Source data versioned with DVC

The original Inside files (`GBG500.parquet`, `GBG500_labels.csv`, `data.txt`) live in
`data/source/` and are tracked by DVC (`data/source.dvc`, ~29 MB). Git only stores the
fingerprint; the files are in the DVC remote `storage` = `s3://datasets/dvc` on MinIO
(`http://localhost:9000`). `SRV3490.parquet` is not tracked (not required).

```bash
export AWS_ACCESS_KEY_ID=<MINIO_ROOT_USER> AWS_SECRET_ACCESS_KEY=<MINIO_ROOT_PASSWORD>
uv run dvc pull                                  # download the data
uv run dvc add data/source && uv run dvc push    # after changing the data
```

MinIO runs locally, so each member first pushes the data to their own MinIO
(copy the Inside files into `data/source/`, then `uv run dvc push`).

## From rides to images

`src/trajectory_mlops/data/convert.py` (DVC stage `convert`) turns each ride into
grayscale images saved in `data/raw/<CLASS>/<RIDE_ID>_w<NN>.png`:

- each ride is cut into 30 s windows (300 samples at 10 Hz); the incomplete tail is dropped;
- one horizontal band per signal (`ax ay az rx ry rz`), one pixel column per 0.1 s;
- light = low value, dark = high value, with the **same fixed range for every image**
  (`convert.ranges` in `params.yaml`), so a strong shake looks dark in every ride;
- missing sensor rows are filled from neighbouring values.

Result: 4,849 images (Safe 4,021, Safe Overall 375, Bad Weather 226, Tandem 160,
Reckless 67). `data/raw/` is a DVC output: regenerate it with `uv run dvc repro convert`.
Ne jamais utiliser un chemin `C:/Users/...` comme remote pour tout le groupe.
Ne pas publier les données Inside sur GitHub sans autorisation.

