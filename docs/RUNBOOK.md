# Operations and data

## Start, inspect, stop

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail 100 mlflow api
docker compose stop
```

Storage survives a stop thanks to the volumes: keep them to keep the models and the
metadata. The API can start without a model; its readiness stays at 503 until a valid
model is configured and the API is restarted.

## Data versioning with DVC and MinIO

The source data (`data/source/`) is already tracked with DVC (`data/source.dvc`) and the
default remote `storage` (`s3://datasets/dvc`, endpoint `http://localhost:9000`) is
configured in `.dvc/config`. With MinIO running (`docker compose up -d minio init-bucket`):

```bash
export AWS_ACCESS_KEY_ID=<MINIO_ROOT_USER> AWS_SECRET_ACCESS_KEY=<MINIO_ROOT_PASSWORD>
uv run dvc pull                 # download the data recorded in data/source.dvc
uv run dvc repro                # rerun the DAG; dvc.lock records the versions
uv run dvc push                 # upload new DVC outputs (images, model) to MinIO
```

After changing the source data:

```bash
uv run dvc add data/source
uv run dvc push
git add data/source.dvc
git commit -m "Update source data"
```

Never commit credentials. `localhost` is each member's own machine: MinIO runs locally,
so each member first copies the Inside files into `data/source/` and runs `dvc push`
to their own MinIO. To share one remote, point the endpoint at a MinIO or S3 reachable
by the whole group with `uv run dvc remote modify --local storage endpointurl <URL>`.

From a fresh clone: `uv sync --frozen --extra ml --extra pipeline --extra dev`, set the
credentials, then `uv run dvc pull`. To restore an older version: `git checkout TAG`,
then `uv run dvc pull` to get the matching bytes.

The DVC cache can restore references to an old run, but reproducing the files does not
restore a deleted MLflow server. To create a new run in an empty registry:
`uv run dvc repro --force` with the right tracking URI.

## Common errors

- `Permission denied` on mlflow/dvc: use `uv run python -m mlflow` or
  `uv run python -m dvc`, as in the class TP.
- Readiness 503: check the logs, `MODEL_URI`, the version/alias and the target server.
- MLflow unreachable: check local port 5001, the allowed hosts and the MinIO service.
- `Unable to locate credentials` from DVC: export `AWS_ACCESS_KEY_ID` and
  `AWS_SECRET_ACCESS_KEY` in the current terminal.
- Promotion refused: read the report; the model must beat the baselines and the
  current `@production` model.
- DVC missing files: the data owner must have run `dvc push`.
- Docker unavailable: start Docker Desktop (Linux containers).

## Reproduction

Keep the Git commit, `uv.lock`, `dvc.lock`, the dataset version, the parameters, the
CPU/GPU environment and the run ID. Seeds are fixed for the CPU prototype; exact
numerical equality across different hardware or library versions is not guaranteed.

## Rollback

Pick an earlier evaluated version in the target registry. Set `MODEL_URI` to
`models:/trajectory-classifier/NUMBER`, then recreate the Compose API or trigger the
Kubernetes deployment. Check the version returned by `/predict` and rerun the smoke test.
