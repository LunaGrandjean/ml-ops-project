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

The promotion gate requires test accuracy above uniform random guessing (20%),
test accuracy of at least the configured threshold (21%), and test macro-F1 above
the majority-class baseline. When `@production` exists, promotion also requires
a matching dataset/split fingerprint and a strictly higher macro-F1. Use
`--dry-run` to preview the decision, or `--output artifacts/promotion.json`
to save it. A skipped promotion exits successfully with `promoted=false`;
automation must check that flag before releasing. See the
[promotion policy](docs/RETRAINING.md) for details.

Dans `.env`, renseigner :

```dotenv
MODEL_URI=models:/trajectory-classifier@production
```

Puis recharger l'API (les alias sont résolus au démarrage) :

```powershell
docker compose up -d --force-recreate api ui
uv run python scripts/smoke_api.py --image "data/raw/Safe/exemple.png"
```

## Model architecture and evaluation

Each example is a 30-second window of accelerometer and gyroscope signals
encoded as an image with six horizontal bands. The compact PyTorch CNN uses
three convolutional blocks to detect local patterns such as sudden signal
changes. Pooling reduces sensitivity to where a pattern occurs in the window,
and a linear layer predicts one of the five classes. This architecture can be
trained from scratch on CPU and exported as TorchScript.

The split groups windows by ride to prevent leakage. Training uses inverse-frequency
class weights to give rare classes more influence and applies random time shifts
and small amounts of noise to training images. Checkpoints are selected by
validation macro-F1, which gives equal weight to each class.

### Reported experiments

| Experiment | Configuration | Test accuracy | Test macro-F1 | Reckless recall |
| --- | --- | ---: | ---: | ---: |
| 1 | CNN, 10 epochs | 0.845 | 0.268 | 4/15 (27%) |
| 2 | CNN with class weights | 0.674 | 0.319 | 11/15 (73%) |
| 3 | Class weights and augmentation | 0.692 | 0.343 | 12/15 (80%) |

The majority classifier reached 0.842 test accuracy. Experiment 3 had the highest
reported macro-F1. Bad Weather and Tandem still had zero recall. The
[EDA](docs/EDA.md) shows that these classes have few rides and motion profiles
similar to Safe. Reckless recall is measured on windows from a single test ride,
which limits what the result says about performance on unseen Reckless rides.

[Experiment 1 confusion matrix](docs/screenshots/confusion_matrix_experiment1.png) ·
[Experiment 2 confusion matrix](docs/screenshots/confusion_matrix_experiment2.png) ·
[Experiment 3 confusion matrix](docs/screenshots/confusion_matrix_experiment3.png)

### Champion comparison and tuning

The saved confusion matrices for experiments 2 and 3 contain the same class
supports across 736 test windows. They show the following comparison.

| Test class | Support | v2 recall | v3 recall |
| --- | ---: | ---: | ---: |
| Safe | 620 | 462/620 (74.5%) | 473/620 (76.3%) |
| Safe Overall | 48 | 23/48 (47.9%) | 24/48 (50.0%) |
| Bad Weather | 24 | 0/24 (0%) | 0/24 (0%) |
| Tandem | 29 | 0/29 (0%) | 0/29 (0%) |
| Reckless | 15 | 11/15 (73.3%) | 12/15 (80.0%) |

On these saved results, v3 improved macro-F1 by 0.024 and made 13 more correct
predictions. Its reported accuracy and macro-F1 exceed the promotion thresholds
(0.210 accuracy and approximately 0.183 majority-class macro-F1).

The last team report recorded v2 as `@production` and v3 as `@candidate`.
The source directory and regenerated manifest matched the DVC hashes in
`dvc.lock`. Tuning kept the data split, seed, class weights and augmentation
fixed and varied the epoch budget and Adam learning rate.

| Epoch budget | Learning rate | Best epoch | Validation macro-F1 | Local MLflow run |
| ---: | ---: | ---: | ---: | --- |
| 10 | 0.001 | 10 | 0.3515 | First 10 epochs of the 0.001 run |
| 20 | 0.001 | 18 | 0.3663 | `fd03289cbff7485db28760c2bd4dff09` |
| 20 | 0.0005 | 18 | 0.3678 | `162082769cfd448e8c6d1ea6080e83ed` |

The checkpoint selected on validation (0.0005, epoch 18) achieved **0.3597 test
macro-F1** and **0.7514 accuracy** on 736 windows. Its recall was 84.2% for
Safe (522/620), 41.7% for Safe Overall (20/48), 0% for Bad Weather (0/24),
0% for Tandem (0/29), and 73.3% for Reckless (11/15). Compared with v3,
the tuned model improved macro-F1 by about 0.017, while recall declined for
Safe Overall and Reckless. [Tuned model confusion matrix](docs/screenshots/confusion_matrix_tuned.png)

The tuned model passed the quality gate and was promoted as v1 in the local
MLflow registry. The deployment team can register the evaluated bundle in the Kubernetes MLflow
registry without retraining. See [Person 2 model handoff](docs/DEPLOYMENT.md#person-2-model-handoff).

## Tests

```powershell
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m pytest -m "not ml and not e2e"
uv run python -m pytest -m ml
```

ML integration tests cover a tiny synthetic training pipeline and import/HTTP serving
of the actual evaluated handoff model in an isolated SQLite registry. They do not
change the team registry or claim new project accuracy results. Pour tester le service
déployé avec une véritable image :

```powershell
$env:E2E_API_URL = "http://localhost:8000"
$env:E2E_IMAGE_PATH = "data/raw/Safe/exemple.png"
uv run python -m pytest -m e2e
```

## Kubernetes, CI/CD et collaboration

- Dev : Docker Compose. Production pédagogique : cluster Kubernetes local.
- Publication des trois images sur Docker Hub au tag Git `v*` ou lancement manuel.
- CD vers Kubernetes activée avec `ENABLE_K8S_DEPLOY=true`, un runner Mac de confiance
  (labels configurables via MLOPS_RUNNER_LABELS) et l'environnement GitHub `production`.
- Le registre MLflow du cluster est distinct du registre Compose. Un modèle doit
  être enregistré dans le registre cible avant que l'API puisse le charger.
- Procédure, secrets et première initialisation : [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
- Règles de PR : [CONTRIBUTING.md](CONTRIBUTING.md).
- API metrics are implemented; Prometheus/Grafana infrastructure remains with Member 5.
- Lightweight feature store remains an optional Member 1 bonus.

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
| Model experiments and majority baseline | See [reported experiments](#reported-experiments) |
| Final registry version and comparison | Tuned model v1 in local MLflow. Team registry pending. |
| Trois URLs Docker Hub | À renseigner |
| Captures Kubernetes / Swagger / UI / MLflow | À ajouter dans docs/screenshots/ |

## Backend and retraining

- [API and Streamlit guide](docs/BACKEND.md): readiness, model metadata, upload
  validation, five confidence scores, error handling and demo commands.
- [Retraining and promotion](docs/RETRAINING.md): manual dry run, DVC retrieval,
  artifact smoke check, comparable champion/challenger gate and automatic release
  only for an improved candidate. Releases pin both code SHA and numbered model URI.
- [API monitoring handoff](docs/API_MONITORING.md): Prometheus endpoint, metric
  names, scrape configuration and Grafana query examples for Member 5.

Code and isolated tests are provided. Running the release still requires the
team's Docker Hub secrets, trusted runner, production MLflow/MinIO and Kubernetes.
