"""Compare an evaluated candidate with production before moving the alias."""

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from trajectory_mlops.config import Settings


def _score(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if isfinite(value) and 0 <= value <= 1 else None


def quality_gate(metrics: dict[str, float]) -> bool:
    """Preserve the team's macro-F1 gate; reject missing or invalid scores."""
    keys = (
        "test_accuracy",
        "random_accuracy",
        "min_accuracy",
        "test_macro_f1",
        "baseline_macro_f1",
    )
    values = {key: _score(metrics, key) for key in keys}
    if any(value is None for value in values.values()):
        return False
    return (
        values["test_accuracy"] > values["random_accuracy"]
        and values["test_accuracy"] >= values["min_accuracy"]
        and values["test_macro_f1"] > values["baseline_macro_f1"]
    )


@dataclass(frozen=True)
class PromotionResult:
    candidate_version: str
    previous_version: str | None
    eligible: bool
    promoted: bool
    reason: str
    model_uri: str


def _production_version(client, name: str) -> str | None:
    # Inspect aliases instead of catching "alias not found": MLflow backends use
    # different error codes for that case. Authentication/network errors must propagate.
    version = client.get_registered_model(name).aliases.get("production")
    return str(version) if version is not None else None


def promote(
    version: str,
    *,
    dry_run: bool = False,
    client=None,
    settings: Settings | None = None,
) -> PromotionResult:
    """Promote only an improved, comparable candidate; skips are normal decisions.

    Client injection lets unit tests verify that rejected candidates never mutate
    the registry. Real clients are created only when this function is called.
    """
    settings = settings or Settings()
    if client is None:
        from mlflow import MlflowClient

        client = MlflowClient(
            tracking_uri=settings.mlflow_tracking_uri, registry_uri=settings.mlflow_tracking_uri
        )

    name = settings.model_name
    candidate = client.get_model_version(name, version)
    version = str(candidate.version)
    previous = _production_version(client, name)
    model_uri = f"models:/{name}/{version}"

    def result(eligible: bool, reason: str, promoted: bool = False) -> PromotionResult:
        return PromotionResult(version, previous, eligible, promoted, reason, model_uri)

    if version == previous:
        return result(False, "Candidate is already in production.")
    if candidate.status != "READY" or not candidate.run_id:
        return result(False, "Candidate must be READY and linked to an evaluated run.")

    run = client.get_run(candidate.run_id)
    if run.info.status != "FINISHED":
        return result(False, "Candidate evaluation run must be FINISHED.")
    if not quality_gate(run.data.metrics):
        return result(False, "Candidate fails the baseline quality gate.")

    fingerprint = run.data.tags.get("data_sha256", "").strip()
    if not fingerprint:
        return result(False, "Candidate has no dataset/split fingerprint.")

    if previous is not None:
        champion = client.get_model_version(name, previous)
        if not champion.run_id:
            return result(False, "Production model has no evaluation run to compare.")
        champion_run = client.get_run(champion.run_id)
        if champion_run.info.status != "FINISHED":
            return result(False, "Production evaluation run must be FINISHED.")
        champion_fingerprint = champion_run.data.tags.get("data_sha256", "").strip()
        if fingerprint != champion_fingerprint:
            return result(
                False, "Dataset/split fingerprints differ; re-evaluate on the same split."
            )
        champion_f1 = _score(champion_run.data.metrics, "test_macro_f1")
        if champion_f1 is None:
            return result(False, "Production macro-F1 is missing or invalid.")
        if float(run.data.metrics["test_macro_f1"]) <= champion_f1:
            return result(False, "Candidate macro-F1 does not improve on production.")

    if dry_run:
        return result(True, "Candidate is eligible; dry run leaves production unchanged.")

    # Detect changes since the comparison. This is not an atomic compare-and-swap;
    # automated promotion jobs must also share a registry-level concurrency group.
    if _production_version(client, name) != previous:
        return result(False, "Production changed during comparison; retry the decision.")
    client.set_registered_model_alias(name, "production", version)
    return result(True, "Candidate promoted to production.", promoted=True)


def main(argv: list[str] | None = None) -> PromotionResult:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Compare without moving the alias")
    parser.add_argument("--output", type=Path, help="Write the JSON decision for workflow steps")
    args = parser.parse_args(argv)
    decision = promote(args.version, dry_run=args.dry_run)
    payload = json.dumps(asdict(decision), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return decision


if __name__ == "__main__":
    main()
