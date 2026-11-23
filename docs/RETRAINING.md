# Retraining: promotion policy

The manual **Retrain and release** workflow retrieves the DVC source data, runs
the complete DAG, verifies the candidate can serve predictions, compares it with
production, and calls the release workflow only after an actual promotion.

## Policy

An evaluated candidate must:
1. Be a READY registered version linked to a FINISHED MLflow run.
2. Pass the existing team gate: accuracy above uniform random guessing, accuracy
   at least the logged minimum, and macro-F1 above the majority baseline.
3. Have a non-empty `data_sha256` run tag.
4. If `@production` exists, have the same dataset/split fingerprint as its
   FINISHED evaluation run and a **strictly higher test macro-F1**.

Missing, non-finite or out-of-range scores block promotion. Equal scores do not
replace production. Accuracy is not required to beat the majority classifier:
the team deliberately uses macro-F1 for its imbalanced classes.

The fingerprint is the manifest hash logged by training and preserved by the
model handoff. If datasets/splits differ, re-evaluate both models on one shared,
fixed comparison set and log its fingerprint before comparing scores. Never
silently compare metrics from different evaluation populations.

This uses the currently logged test metrics for compatibility with the team's
existing evaluated models. Repeated automated model selection can bias that test
set: keep a separate untouched final holdout for the final project assessment.

## Commands

First preview the decision using the tracking URI for the intended registry:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
uv run python -m trajectory_mlops.promote --version 2 --dry-run
```

To apply an eligible decision and save a machine-readable report:

```powershell
uv run python -m trajectory_mlops.promote --version 2 --output artifacts/promotion.json
```

The JSON includes `eligible`, `promoted`, `reason`, `candidate_version`,
`previous_version` and the immutable candidate `model_uri`.

- Dry run: an eligible candidate has `eligible=true`, `promoted=false`.
- Rejected candidate or already-production version: `promoted=false`, exit code 0.
- Successful alias update: `promoted=true`, exit code 0.
- Registry/authentication/network errors: command fails; no success report is emitted.

The next workflow step must trigger a release **only when promoted is true**,
using the returned numbered model URI. A passing exit code alone is insufficient.
Remove any prior report before starting a new CI attempt so that a failed command
cannot leave stale output looking like a fresh decision.

Production is read again immediately before the alias update. If it changed,
promotion is skipped. This is not an atomic registry transaction: serialize all
automated promotion writers through one shared GitHub concurrency group.
Manual alias changes must also be coordinated.

API processes resolve the model at startup. Moving an alias alone does not reload
a running API; deployment/restart and a real inference check are still required.
For an explicit rollback, configure a previously evaluated numbered MODEL_URI and
redeploy; normal promotion intentionally rejects a lower macro-F1.

## Verification

Unit tests exercise baseline failures, missing/NaN/invalid metrics, split
mismatches, bootstrap, dry run, concurrency detection and registry failures.
The registry integration test uses a temporary SQLite backend and placeholder
artifact directories: it validates MLflow aliases without retraining a model or
touching the team's running registry.

## Workflow setup and operation

Configure the repository variables with Member 5:

| Variable | Value |
| --- | --- |
| MLOPS_RUNNER_LABELS | JSON array, default `["self-hosted","macOS","mlops"]` |
| MLFLOW_TRAINING_URI | Reachable MLflow URL backed by the **deployed production registry** |
| DATA_S3_ENDPOINT | Reachable MinIO S3 URL for the versioned DVC dataset |
| MODEL_NAME | Optional, defaults to `trajectory-classifier` |
| ENABLE_K8S_DEPLOY | `true` only when the runner and cluster are ready |
| MODEL_URI | Numbered URI for manual/tagged releases; automated releases use the decision |

Create environments `training` and `production`. Configure MINIO_ROOT_USER,
MINIO_ROOT_PASSWORD, DOCKERHUB_USERNAME and DOCKERHUB_TOKEN secrets. The local
runner needs bash, Docker, kubectl, Python 3 and cluster access. Restrict these
workflows/environments to trusted branches; PR CI uses hosted runners.

Start **Retrain and release** from Actions with `dry_run=true` first (the default).
Download the candidate report artifact and inspect the decision, evaluation,
confusion matrix and DVC lock. Dry run trains/registers a candidate but never
moves production or calls release. Retry with `dry_run=false` to enable promotion.
The workflow fails early if deployment is disabled for a non-dry run.

The output is tied to this run's registry version and run ID; stale reports are
removed. Broken candidate artifacts fail a prediction smoke check before any
alias mutation. A rejection is a successful decision, with no release job.

Retraining holds the `production` concurrency group through its nested release;
standalone releases use that same group. The nested release uses its own run
identifier to avoid waiting on its caller's lock. It verifies the source, builds
all three images at the exact commit SHA, deploys the numbered model URI, then
checks a real HTTP prediction and its returned version inside the cluster.
The generated white-image smoke input verifies serving, not classification quality.

Training must target the same registry used by the deployed API. A version number
from a separate development registry is not transferable. Use the handoff import
procedure for initial bootstrap. New data or a changed split will deliberately
fail the fingerprint comparison until both candidates are evaluated comparably.

If deployment fails after promotion, the old API can continue serving its pinned
version while the alias points to the new one. Inspect the failed release, restore
the previous numbered URI and redeploy, or repair and rerun the release with the
new numbered URI. The workflow never automatically hides rollout failures.

Workflow syntax follows [GitHub reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows).
