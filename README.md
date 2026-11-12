# Trajectory MLOps — projet de groupe

Socle de projet pour classifier des **images de trajectoires** en cinq catégories :
**Safe, Safe Overall, Bad Weather, Tandem, Reckless**.

Ce dossier est la **racine du futur dépôt GitHub**. Publier son contenu à la racine
du dépôt pour que `.github/workflows/` soit reconnu par GitHub Actions.

Le pipeline et les services sont implémentés comme point de départ. Les données
Inside, l'entraînement réel, les résultats, les cinq noms du groupe, les identifiants
Docker Hub et le déploiement du cluster restent à renseigner et à vérifier.
Aucune performance sur les données réelles n'est revendiquée.

## Dates et livrables

- **20/11/2026** : transmettre les cinq membres aux enseignants.
- **25/11/2026 à 20 h, heure de Paris** : remise, pénalités de retard.
- **26/11/2026** : présentation de 10 min puis questions de 10 min.
- Consignes de rendu : [docs/DELIVERY.md](docs/DELIVERY.md).
- Répartition et étapes : [docs/TEAM.md](docs/TEAM.md).
- Architecture et DAG : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Arborescence

```text
project/                     # racine du futur dépôt
├── src/trajectory_mlops/
│   ├── data/                # validation, SHA-256, split, prétraitement partagé
│   ├── models/              # CNN PyTorch, entraînement et évaluation
│   ├── api/                 # FastAPI, upload d'image, santé, Swagger
│   ├── config.py            # classes et configuration centralisées
│   └── promote.py           # promotion explicite après contrôle des métriques
├── app/                     # interface Streamlit qui appelle l'API
├── data/{raw,processed}/    # données ignorées par Git, versionnables par DVC
├── artifacts/               # modèle local et identifiants de run/version
├── reports/                 # métriques et matrice de confusion générées
├── tests/{unit,integration,e2e}/
├── scripts/                 # rendu K8s et test d'une API réelle
├── k8s/                     # services, stockage persistant, probes
├── .github/workflows/       # CI, images Docker Hub, CD et réentraînement
├── docs/                    # architecture, exploitation, équipe, rendu
├── compose.yaml             # environnement dev
├── Dockerfile               # cibles api, ui, tracking
├── dvc.yaml                 # DAG prepare → train → evaluate
├── params.yaml              # configuration versionnée des expériences
├── pyproject.toml
└── uv.lock                  # dépendances résolues, à committer
```

## Démarrage local — PowerShell

Prérequis : Git, uv, Docker Desktop démarré en mode conteneurs Linux.
Les commandes ci-dessous sont à lancer **dans ce dossier**.

```powershell
Copy-Item .env.example .env
uv sync --frozen --extra ml --extra pipeline --extra dev
docker compose up -d --build
```

Services :
- Swagger : http://localhost:8000/docs
- Interface de test : http://localhost:8501
- MLflow : http://localhost:5001 (pour éviter le port 5000 du TP)
- MinIO : http://localhost:9001 (identifiants dans votre `.env`)

Avant entraînement, `/health/live` répond 200 et `/health/ready` répond 503.
L'interface indique que le modèle n'est pas chargé si vous demandez une prédiction.

## Données et pipeline

1. Télécharger les données Inside et confirmer leur format : [data/README.md](data/README.md).
2. Placer les images dans les cinq sous-dossiers de classes attendus.
3. Avant de travailler, vérifier que le code et les paramètres sont committés.
4. Initialiser DVC une seule fois **après avoir initialisé le dépôt Git** :

```powershell
uv run python -m dvc init
uv run python -m dvc repro
uv run python -m dvc dag
```

Le DAG prépare un manifeste déterministe, entraîne le CNN sur train, choisit le
meilleur epoch sur validation puis évalue sur test. MLflow trace les paramètres,
les métriques, le commit et le SHA-256 du manifeste. Les modèles et artefacts MLflow
passent par le serveur vers le bucket MinIO `mlflow-artifacts`.

Les résultats se trouvent dans `reports/evaluation.json`, le run dans
`artifacts/run.json`, la version enregistrée dans `artifacts/registry.json`.
Le registre reçoit l'alias `candidate` ; il ne change pas automatiquement la production.

Pour changer les hyperparamètres, modifier `params.yaml`, committer puis relancer
`uv run python -m dvc repro`. Pour reconstituer les mêmes données, suivre
[docs/RUNBOOK.md](docs/RUNBOOK.md) et versionner les entrées avec DVC.

## Promouvoir et servir le modèle

Lire la version dans `artifacts/registry.json`, puis remplacer `1` ci-dessous :

```powershell
uv run python -m trajectory_mlops.promote --version 1
```

Le contrôle exige une accuracy supérieure au hasard uniforme (20 %) et au
classifieur majoritaire, au moins le seuil configuré, et une macro-F1 supérieure
à la baseline majoritaire. Si le contrôle échoue, améliorer le modèle sur validation
avant de préparer l'évaluation finale ; ne pas diminuer les seuils pour masquer l'échec.

Dans `.env`, renseigner :

```dotenv
MODEL_URI=models:/trajectory-classifier@production
```

Puis recharger l'API (les alias sont résolus au démarrage) :

```powershell
docker compose up -d --force-recreate api ui
uv run python scripts/smoke_api.py --image "data/raw/Safe/exemple.png"
```

Le CNN est une baseline compacte, sans téléchargement de poids. Le responsable
modèle pourra le remplacer par du transfert d'apprentissage après avoir confirmé
le dataset et établi les baselines. L'accuracy réelle reste à mesurer.

## Tests

```powershell
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m pytest -m "not ml and not e2e"
uv run python -m pytest -m ml
```

Le test ML utilise un dataset synthétique minuscule et un backend MLflow temporaire :
il vérifie les branchements, pas la qualité du modèle réel. Pour tester le service
déployé avec une véritable image :

```powershell
$env:E2E_API_URL = "http://localhost:8000"
$env:E2E_IMAGE_PATH = "data/raw/Safe/exemple.png"
uv run python -m pytest -m e2e
```

## Kubernetes, CI/CD et collaboration

- Dev : Docker Compose. Production pédagogique : cluster Kubernetes local.
- Publication des trois images sur Docker Hub au tag Git `v*` ou lancement manuel.
- CD vers Kubernetes activée avec `ENABLE_K8S_DEPLOY=true`, un runner Linux
  auto-hébergé de confiance et l'environnement GitHub `production`.
- Le registre MLflow du cluster est distinct du registre Compose. Un modèle doit
  être entraîné/enregistré dans le registre cible avant de déployer l'API.
- Procédure, secrets et première initialisation : [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
- Règles de PR : [CONTRIBUTING.md](CONTRIBUTING.md).
- Observabilité avancée et feature store : backlog bonus, non implémentés.

## Publication du dossier

Créer un **nouveau dépôt vide** pour le projet de groupe, puis, depuis ce dossier :

```powershell
git init -b main
git add .
git commit -m "Initial project architecture"
git remote add origin https://github.com/TON_COMPTE/TON_NOUVEAU_DEPOT.git
git push -u origin main
```

Ne pas reprendre le dépôt du TP en remplaçant son historique. Inviter les quatre
autres membres comme collaborateurs et activer une revue de PR sur `main`.

## Résultats à compléter avant rendu

| Élément | Valeur |
| --- | --- |
| Version du dataset / DVC | À renseigner |
| Split et prévention des fuites | À confirmer avec les données Inside |
| Baseline majoritaire accuracy / macro-F1 | À mesurer |
| Meilleur run MLflow / version | À mesurer |
| Modèle final accuracy / macro-F1 | À mesurer |
| Trois URLs Docker Hub | À renseigner |
| Captures Kubernetes / Swagger / UI / MLflow | À ajouter dans docs/screenshots/ |
