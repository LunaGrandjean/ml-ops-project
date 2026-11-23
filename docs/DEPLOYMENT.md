# Déploiement Kubernetes et GitHub Actions

Prérequis : Docker Desktop Kubernetes activé, kubectl sur le bon contexte,
stockage dynamique disponible pour les PVC, images Docker Hub publiques ou
imagePullSecrets configurés. Les manifests sont destinés à un cluster local.

## Première initialisation du cluster

1. Construire et publier les trois images api, ui, tracking avec un même tag.
   Le workflow release les construit après vérification, avec le SHA du commit.
2. Créer le namespace et le secret puis le stockage :

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl -n trajectory-mlops create secret generic storage-credentials --from-literal=MINIO_ROOT_USER=VOTRE_USER --from-literal=MINIO_ROOT_PASSWORD=VOTRE_PASSWORD
kubectl apply -f k8s/storage.yaml
kubectl -n trajectory-mlops rollout status deployment/minio
kubectl apply -f k8s/init-buckets.yaml
kubectl -n trajectory-mlops wait --for=condition=complete job/init-buckets --timeout=180s
```

3. Rendre puis appliquer les apps. Choisir un numéro de modèle existant pour les
   déploiements reproductibles ; l'exemple de bootstrap utilise l'alias production :

```powershell
uv run python scripts/render_k8s.py --registry TON_COMPTE_DOCKERHUB --tag TON_SHA --model-uri models:/trajectory-classifier@production
kubectl apply -f k8s/rendered/apps.yaml
kubectl -n trajectory-mlops rollout status deployment/mlflow --timeout=300s
# Libérer le port dev avant le port-forward du registre de production.
docker compose stop mlflow
kubectl -n trajectory-mlops port-forward service/mlflow 5001:5000
```

4. **Bootstrap modèle** : le registre du cluster est initialement vide. L'API ne sera
   pas ready et c'est attendu. Garder le port-forward MLflow dans un terminal,
   puis suivre [Person 2 model handoff](#person-2-model-handoff). Après promotion,
   redémarrer l'API pour qu'elle résolve l'alias :

```powershell
kubectl -n trajectory-mlops rollout restart deployment/api
kubectl -n trajectory-mlops rollout status deployment/api --timeout=300s
```

5. Ouvrir dans deux autres terminaux :

```powershell
kubectl -n trajectory-mlops port-forward service/api 8000:8000
kubectl -n trajectory-mlops port-forward service/ui 8501:8501
```

Tester Swagger, la prédiction par l'UI et `scripts/smoke_api.py`. Garder des captures
du cluster, du bucket, du run et de la version du modèle servi.

## Person 2 model handoff

Download [person2-model-handoff.zip](handoff/person2-model-handoff.zip) for the MLflow model, training
configuration, source run ID, dataset fingerprint and test evaluation.
Extract it on a machine that can reach the target MLflow server. With the
cluster MLflow port-forward active, register the bundle:

```bash
uv sync --frozen --extra tracking
uv run python scripts/register_model_bundle.py /path/to/person2-model-handoff --tracking-uri http://localhost:5001
```

The script prints the version assigned by the target registry. If it already
has a production model, compare the runs on the same test split and confirm
the model choice before moving the alias. Replace `TARGET_VERSION` with the
version printed above:

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 uv run python -m trajectory_mlops.promote --version TARGET_VERSION
```

The API uses `MODEL_URI=models:/trajectory-classifier@production`. Restart it
after promotion and check readiness, the returned `model_version`, and a
prediction from a real GBG500 window. The MLflow server stores the model in the
cluster's MinIO bucket through its artifact store.

## Configuration GitHub

Secrets à créer :
- DOCKERHUB_USERNAME / DOCKERHUB_TOKEN : token autorisé à publier les images.
- MINIO_ROOT_USER / MINIO_ROOT_PASSWORD : stockage du cluster.

Variables :
- ENABLE_K8S_DEPLOY : `true` uniquement lorsque le runner et le modèle cible sont prêts.
- MODEL_URI : `models:/trajectory-classifier/NUMERO` recommandé en production.
- MLFLOW_TRAINING_URI : URL du MLflow cible accessible au runner.
- DATA_S3_ENDPOINT : URL S3 des données accessible au runner.

Create GitHub environments `production` and `training`. The default runner labels
are `self-hosted, macOS, mlops` for the team's Mac with Docker Desktop Kubernetes.
Override `MLOPS_RUNNER_LABELS` with a JSON array for another operating system.
Install bash, Docker, kubectl and Python 3, and configure the runner's kubeconfig.
Hosted GitHub runners cannot reach a local cluster. Keep PR jobs hosted and
reserve local runners for trusted workflows. See [retraining](RETRAINING.md).

## Workflows

- ci.yml : lint, tests, intégration ML synthétique et build des trois images.
- release.yml : vérifie, publie les images taguées au SHA puis déploie si activé.
- retrain.yml: manual DVC pull and full DAG, candidate artifact smoke check,
  guarded promotion, then reusable release only when promoted. Dry run is the
  default. Requires the committed `data/source.dvc` pointer and configured remote.

Déclencher une livraison avec un tag, par exemple `v0.1.0`, après configuration
des secrets. Le déploiement utilise des images identifiées par SHA, sans écraser latest.

## Vérifications et limites

- Un premier déploiement sans modèle n'est pas un déploiement d'inférence terminé.
- SQLite MLflow : une réplique et stratégie Recreate pour le volume persistant.
- Sur échec de chargement, l'API reste live mais non ready ; corriger la config,
  vérifier les services puis redémarrer le déploiement.
- CD checks readiness, an HTTP prediction and the served model version using a
  generated image. Also run the smoke command with an authorized GBG500 window
  for the project demonstration; generated images do not measure model quality.
- Exposer les services par port-forward pour la démonstration locale ; pas
  d'authentification Internet ou d'Ingress public dans ce socle.
