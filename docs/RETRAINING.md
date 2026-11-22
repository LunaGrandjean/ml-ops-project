# Retraining: promotion policy

The first implementation step adds a safe champion/challenger decision to
`trajectory_mlops.promote`. The GitHub retraining/release workflow wiring is a
separate step; this change does not claim that automatic retraining is deployed.

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
