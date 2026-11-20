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

## Données sources versionnées avec DVC

Les fichiers Inside d'origine (`GBG500.parquet`, `GBG500_labels.csv`, `data.txt`)
sont dans `data/source/`, suivis par DVC (`data/source.dvc`, ~29 Mo). Git ne stocke
que l'empreinte ; les fichiers sont dans le remote `storage` = `s3://datasets/dvc`
sur MinIO (`http://localhost:9000`). `SRV3490.parquet` n'est pas suivi (non requis).

```bash
export AWS_ACCESS_KEY_ID=<MINIO_ROOT_USER> AWS_SECRET_ACCESS_KEY=<MINIO_ROOT_PASSWORD>
uv run dvc pull      # récupérer les données
uv run dvc add data/source && uv run dvc push   # après modification des données
```

MinIO tourne en local : chacun doit d'abord pousser les données sur son propre
MinIO (copier les fichiers Inside dans `data/source/` puis `uv run dvc push`).
`data/raw/` reste réservé aux images générées à partir de `data/source/`.
Ne jamais utiliser un chemin `C:/Users/...` comme remote pour tout le groupe.
Ne pas publier les données Inside sur GitHub sans autorisation.

