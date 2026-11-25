# Kubernetes deployment and GitHub Actions

Requirements: Docker Desktop Kubernetes enabled, kubectl on the right context,
dynamic storage available for the PVCs, and public Docker Hub images (or
imagePullSecrets configured). The manifests target a local cluster.

## First cluster setup

1. Build and publish the three images (api, ui, tracking) with the same tag.
   The release workflow builds them after verification, tagged with the commit SHA.
2. Create the namespace, the secret, then the storage:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl -n trajectory-mlops create secret generic storage-credentials --from-literal=MINIO_ROOT_USER=YOUR_USER --from-literal=MINIO_ROOT_PASSWORD=YOUR_PASSWORD
kubectl apply -f k8s/storage.yaml
kubectl -n trajectory-mlops rollout status deployment/minio
kubectl apply -f k8s/init-buckets.yaml
kubectl -n trajectory-mlops wait --for=condition=complete job/init-buckets --timeout=180s
```

3. Render then apply the apps. Pick an existing model version number for
   reproducible deployments; the bootstrap example uses the production alias:

```bash
uv run python scripts/render_k8s.py --registry YOUR_DOCKERHUB_USERNAME --tag YOUR_COMMIT_SHA --model-uri models:/trajectory-classifier@production
kubectl apply -f k8s/rendered/apps.yaml
kubectl -n trajectory-mlops rollout status deployment/mlflow --timeout=300s
# Free the dev port before port-forwarding the production registry.
docker compose stop mlflow
kubectl -n trajectory-mlops port-forward service/mlflow 5001:5000
```

4. **Model bootstrap**: the cluster registry starts empty, so the API will not be
   ready yet; this is expected. Keep the MLflow port-forward open in one terminal,
   then follow [Person 2 model handoff](#person-2-model-handoff). After promotion,
   restart the API so it resolves the alias:

```bash
kubectl -n trajectory-mlops rollout restart deployment/api
kubectl -n trajectory-mlops rollout status deployment/api --timeout=300s
```

5. Open in two more terminals:

```bash
kubectl -n trajectory-mlops port-forward service/api 8000:8000
kubectl -n trajectory-mlops port-forward service/ui 8501:8501
```

Test Swagger, a prediction from the web app and `scripts/smoke_api.py`. Keep
screenshots of the cluster, the bucket, the run and the served model version.

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

## GitHub configuration

Secrets to create:
- `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`: token allowed to publish the images.
- `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`: cluster storage.

Variables:
- `ENABLE_K8S_DEPLOY`: `true` only once the runner and the target model are ready.
- `MODEL_URI`: `models:/trajectory-classifier/NUMBER` recommended in production.
- `MLFLOW_TRAINING_URI`: URL of the target MLflow server, reachable from the runner.
- `DATA_S3_ENDPOINT`: S3 URL of the data, reachable from the runner.

Create GitHub environments `production` and `training`. The default runner labels
are `self-hosted, macOS, mlops` for the team's Mac with Docker Desktop Kubernetes.
Override `MLOPS_RUNNER_LABELS` with a JSON array for another operating system.
Install bash, Docker, kubectl and Python 3, and configure the runner's kubeconfig.
Hosted GitHub runners cannot reach a local cluster. Keep PR jobs hosted and
reserve local runners for trusted workflows. See [retraining](RETRAINING.md).

## Workflows

- `ci.yml`: lint, tests, synthetic ML integration test and build of the three images.
- `release.yml`: verifies, publishes the images tagged with the commit SHA, then deploys if enabled.
- `retrain.yml`: manual DVC pull and full DAG, candidate artifact smoke check,
  guarded promotion, then reusable release only when promoted. Dry run is the
  default. Requires the committed `data/source.dvc` pointer and configured remote.

Trigger a release with a tag, for example `v0.1.0`, once the secrets are configured.
Deployments use images identified by commit SHA and never overwrite `latest`.

## Checks and limitations

- A first deployment without a model is not a finished inference deployment.
- MLflow on SQLite: one replica and the Recreate strategy for the persistent volume.
- If the model fails to load, the API stays live but not ready; fix the configuration,
  check the services, then restart the deployment.
- CD checks readiness, an HTTP prediction and the served model version using a
  generated image. Also run the smoke command with an authorised GBG500 window
  for the project demonstration; generated images do not measure model quality.
- Services are exposed with port-forward for the local demonstration; there is no
  internet authentication or public Ingress in this setup.
