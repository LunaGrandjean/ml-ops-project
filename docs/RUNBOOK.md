# Exploitation et données

## Démarrer, inspecter, arrêter

```powershell
docker compose up -d --build
docker compose ps
docker compose logs --tail 100 mlflow api
docker compose stop
```

Le stockage survit à l'arrêt grâce aux volumes. Conserver les volumes pour garder
les modèles et les métadonnées. L'API peut démarrer sans modèle ; sa readiness reste
à 503 jusqu'à configuration d'un modèle valide puis redémarrage.

## Versionner les données via MinIO

Après initialisation de Git et DVC et dépôt des images Inside dans data/raw :

```powershell
git rm --ignore-unmatch data/raw/.gitkeep
uv run python -m dvc add data/raw
uv run python -m dvc remote add -d storage s3://datasets/trajectories
uv run python -m dvc remote modify --local storage endpointurl http://localhost:9000
$env:AWS_ACCESS_KEY_ID = "IDENTIFIANT_DE_VOTRE_MINIO"
$env:AWS_SECRET_ACCESS_KEY = "MOT_DE_PASSE_DE_VOTRE_MINIO"
$env:AWS_DEFAULT_REGION = "us-east-1"
uv run python -m dvc push
git add .dvc data/raw.dvc data/.gitignore
git commit -m "Version initial trajectory dataset"
uv run python -m dvc repro
git add dvc.lock
git commit -m "Record pipeline versions"
```

Ne lancer `remote add` qu'une fois. Les identifiants et endpoint locaux ne doivent
pas être versionnés. `localhost` désigne la machine de chaque membre : pour partager
les données, remplacer l'endpoint local par un MinIO accessible au groupe ou un S3
autorisé. Chaque membre configure ce endpoint avec `--local`.

Depuis un clone, installer avec `uv sync --frozen --extra ml --extra pipeline --extra dev`,
configurer le remote et les identifiants, puis `uv run python -m dvc pull`.
Pour restaurer une version : `git checkout TAG`, puis `uv run python -m dvc pull`
pour récupérer les octets manquants et remettre les données à la version du pointeur.

Le cache DVC peut restaurer des références à un ancien run. Reproduire les fichiers
ne restaure pas un serveur MLflow supprimé. Pour créer un nouveau run dans un
registre vide : `uv run python -m dvc repro --force` avec le bon tracking URI.

## Erreurs fréquentes

- `Accès refusé` sur mlflow/dvc : utiliser `uv run python -m mlflow` ou
  `uv run python -m dvc`, comme dans le TP.
- Readiness 503 : consulter les logs, vérifier MODEL_URI, version/alias et serveur cible.
- MLflow inaccessible : vérifier son port local 5001, les allowed-hosts et le service MinIO.
- Promotion refusée : consulter le rapport ; le modèle doit dépasser les baselines.
- DVC missing files : le responsable des données doit avoir fait `dvc push`.
- Docker indisponible : démarrer Docker Desktop et choisir les conteneurs Linux.

## Reproduction

Conserver commit Git, uv.lock, dvc.lock, version du dataset, paramètres, environnement
CPU/GPU et run ID. Les seeds sont fixées pour le prototype CPU ; une égalité numérique
entre matériels/bibliothèques différents n'est pas garantie.

## Rollback

Choisir une version antérieure évaluée dans le registre cible. Définir MODEL_URI
à `models:/trajectory-classifier/NUMERO` puis recréer l'API Compose ou déclencher le
déploiement Kubernetes. Vérifier la version renvoyée par /predict et relancer le smoke test.
