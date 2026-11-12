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
   pas ready et c'est attendu. Garder le
   port-forward MLflow dans un terminal, puis entraîner localement contre le cluster :

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
uv run python -m dvc repro --force
# Lire artifacts/registry.json puis remplacer 1 par la bonne version.
uv run python -m trajectory_mlops.promote --version 1
kubectl -n trajectory-mlops rollout restart deployment/api
kubectl -n trajectory-mlops rollout status deployment/api --timeout=300s
```

Les données doivent être disponibles dans votre checkout. Le serveur MLflow du cluster
envoie les poids à son propre MinIO ; aucune clé S3 n'est nécessaire dans l'API.

5. Ouvrir dans deux autres terminaux :

```powershell
kubectl -n trajectory-mlops port-forward service/api 8000:8000
kubectl -n trajectory-mlops port-forward service/ui 8501:8501
```

Tester Swagger, la prédiction par l'UI et `scripts/smoke_api.py`. Garder des captures
du cluster, du bucket, du run et de la version du modèle servi.

## Configuration GitHub

Secrets à créer :
- DOCKERHUB_USERNAME / DOCKERHUB_TOKEN : token autorisé à publier les images.
- MINIO_ROOT_USER / MINIO_ROOT_PASSWORD : stockage du cluster.

Variables :
- ENABLE_K8S_DEPLOY : `true` uniquement lorsque le runner et le modèle cible sont prêts.
- MODEL_URI : `models:/trajectory-classifier/NUMERO` recommandé en production.
- MLFLOW_TRAINING_URI : URL du MLflow cible accessible au runner.
- DATA_S3_ENDPOINT : URL S3 des données accessible au runner.

Créer les environnements GitHub `production` et `training`. Installer un runner
auto-hébergé **Linux** avec labels `self-hosted, linux, mlops`, docker/kubectl/python3
et un kubeconfig donnant accès au cluster Docker Desktop. Un runner dans WSL/Linux
convient si son accès au cluster est configuré et testé. Un runner GitHub hébergé
ne voit pas votre cluster local. Garder les jobs PR sur les runners hébergés et
réserver le runner local aux workflows de confiance.

## Workflows

- ci.yml : lint, tests, intégration ML synthétique et build des trois images.
- release.yml : vérifie, publie les images taguées au SHA puis déploie si activé.
- retrain.yml : lancement manuel, récupère les données DVC, exécute le DAG,
  enregistre un candidat et conserve les rapports ; promotion ensuite explicite.
  Ce workflow nécessite le remote DVC et le pointeur data/raw.dvc committés.

Déclencher une livraison avec un tag, par exemple `v0.1.0`, après configuration
des secrets. Le déploiement utilise des images identifiées par SHA, sans écraser latest.

## Vérifications et limites

- Un premier déploiement sans modèle n'est pas un déploiement d'inférence terminé.
- SQLite MLflow : une réplique et stratégie Recreate pour le volume persistant.
- Sur échec de chargement, l'API reste live mais non ready ; corriger la config,
  vérifier les services puis redémarrer le déploiement.
- Le smoke test de CD vérifie la readiness ; le test complet d'une image réelle
  reste à exécuter avec les données autorisées.
- Exposer les services par port-forward pour la démonstration locale ; pas
  d'authentification Internet ou d'Ingress public dans ce socle.
