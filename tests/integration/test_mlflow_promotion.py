"""Exercise real registry aliases using an isolated SQLite backend, without training."""

import pytest

pytestmark = pytest.mark.ml


def test_promotion_against_fresh_mlflow_registry(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from trajectory_mlops.config import Settings
    from trajectory_mlops.promote import promote

    monkeypatch.chdir(tmp_path)
    uri = "sqlite:///" + (tmp_path / "registry.db").as_posix()
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    name = "promotion-test"
    client.create_registered_model(name)
    experiment = client.create_experiment("promotion-test", artifact_location=tmp_path.as_uri())
    source = tmp_path / "placeholder-model"
    source.mkdir()
    settings = Settings(model_name=name, mlflow_tracking_uri=uri, _env_file=None)

    def candidate(score):
        run = client.create_run(experiment, tags={"data_sha256": "fixed-evaluation-split"})
        for key, value in {
            "test_accuracy": 0.75,
            "test_macro_f1": score,
            "baseline_macro_f1": 0.18,
            "random_accuracy": 0.2,
            "min_accuracy": 0.21,
        }.items():
            client.log_metric(run.info.run_id, key, value)
        client.set_terminated(run.info.run_id)
        return str(
            client.create_model_version(
                name,
                source=source.as_uri(),
                run_id=run.info.run_id,
            ).version
        )

    first = candidate(0.34)
    assert promote(first, client=client, settings=settings).promoted
    better = candidate(0.36)
    preview = promote(better, client=client, settings=settings, dry_run=True)
    assert preview.eligible and not preview.promoted
    assert str(client.get_model_version_by_alias(name, "production").version) == first
    assert promote(better, client=client, settings=settings).promoted
    worse = candidate(0.3)
    assert not promote(worse, client=client, settings=settings).promoted
    assert str(client.get_model_version_by_alias(name, "production").version) == better
