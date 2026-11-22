# Backend and retraining validation

Validated locally on 22 November 2026 with Python 3.12.15 on Windows, CPU,
MLflow 3.18.0 and PyTorch 2.15.0+cpu, using the locked dependency versions.

- **145 tests passed**, no failures or skips in the isolated suite.
- One deployed-service e2e test was deselected because the team's deployment
  endpoints and authorized image were not configured.
- Ruff lint and formatting passed for all 61 Python files.
- `uv lock --check --offline` passed: 188 packages.
- GitHub workflows passed actionlint 1.7.12 (official release, checksum verified).
- `docker compose config --quiet` passed.
- Git whitespace checks passed.

## What was exercised

- Data conversion, deterministic ride splits, manifests and pipeline consistency.
- Synthetic training on 60 images, temporary SQLite MLflow tracking, evaluation,
  registration and prediction with the reloaded model.
- The actual committed Hoang handoff: import into an isolated registry, normalize
  source registry metadata, load through API startup and perform HTTP inference
  with five valid scores and the expected numbered version.
- Registry-backed and mocked champion/challenger decisions: invalid metrics,
  incompatible splits, bootstrap, dry runs, ties, worse candidates and concurrent
  alias changes. Rejected candidates never move production.
- Retraining candidate identity and artifact smoke checks before promotion;
  stale decision removal and release input validation.
- API readiness, metadata, PNG/JPEG upload, byte/pixel limits, spoofed formats,
  corrupt images, consistent confidence scores and controlled inference errors.
- Streamlit AppTest: empty/unavailable states, upload/classify results and invalid
  images. HTTP client tests cover serialized uploads, timeouts and API errors.
- A real local Uvicorn HTTP connection with the release smoke command, including
  multipart upload, version mismatch rejection and missing-model failure.
- Prometheus request/error counts, latency and confidence histograms, successful
  predictions by class, readiness, bounded labels and per-application isolation.

The synthetic and generated-image tests verify integration, not project accuracy.
The real handoff's existing evaluation is reported in the README; these changes
do not claim a new training result. Test registries and artifacts are temporary;
the team's running MLflow registry and raw data were not modified.

## Remaining environment checks

Docker Desktop's daemon was stopped locally. Container builds, real MLflow/MinIO
S3 transfers, Docker Hub publication and Kubernetes rollout were not run locally.
The temporary SQLite tests do not validate S3 credentials or persistence.

CI now runs on feature pushes and pull requests. It checks lint, formatting,
tests, workflows and Compose, then builds the three container targets. Its remote
status must be checked after pushing; local test success is not a deployment.

With Member 5, configure the trusted runner, repository secrets/variables and the
same production registry for training and serving. Then verify:

1. Compose startup, MinIO artifact persistence and DVC data retrieval.
2. Retraining dry run and its archived evaluation/promotion decision.
3. Eligible promotion, image publication, Kubernetes rollout and in-cluster
   prediction/version smoke check.
4. Streamlit/Swagger with an authorized GBG500 window.
5. Prometheus scraping and Grafana panels, plus a numbered-version rollback.

See [retraining](RETRAINING.md), [backend](BACKEND.md),
[monitoring handoff](API_MONITORING.md) and [deployment](DEPLOYMENT.md).
