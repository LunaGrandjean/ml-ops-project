"""Promotion decisions without network calls or a running MLflow service."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trajectory_mlops.config import Settings
from trajectory_mlops.promote import PromotionResult, promote


def make_run(score=0.36, fingerprint="same-dataset-and-split", status="FINISHED"):
    return SimpleNamespace(
        info=SimpleNamespace(status=status),
        data=SimpleNamespace(
            metrics={
                "test_accuracy": 0.75,
                "test_macro_f1": score,
                "baseline_macro_f1": 0.18,
                "random_accuracy": 0.2,
                "min_accuracy": 0.21,
            },
            tags={} if fingerprint is None else {"data_sha256": fingerprint},
        ),
    )


@pytest.fixture
def registry():
    models = {
        "1": SimpleNamespace(version=1, status="READY", run_id="champion"),
        "2": SimpleNamespace(version=2, status="READY", run_id="candidate"),
    }
    runs = {"champion": make_run(0.34), "candidate": make_run(0.36)}
    client = Mock()
    client.get_model_version.side_effect = lambda name, version: models[str(version)]
    client.get_run.side_effect = lambda run_id: runs[run_id]
    client.get_registered_model.return_value = SimpleNamespace(aliases={"production": 1})
    return client, models, runs


def decide(registry, **kwargs):
    client, _, _ = registry
    return promote(
        "2",
        client=client,
        settings=Settings(model_name="classifier", _env_file=None),
        **kwargs,
    )


def test_better_candidate_replaces_champion(registry):
    decision = decide(registry)
    assert decision.promoted and decision.eligible
    assert decision.previous_version == "1"
    assert decision.model_uri == "models:/classifier/2"
    registry[0].set_registered_model_alias.assert_called_once_with("classifier", "production", "2")


@pytest.mark.parametrize("score", [0.34, 0.3, float("nan"), float("inf")])
def test_non_improving_or_invalid_candidate_never_changes_alias(registry, score):
    registry[2]["candidate"].data.metrics["test_macro_f1"] = score
    assert not decide(registry).promoted
    registry[0].set_registered_model_alias.assert_not_called()


def test_first_production_model_still_has_to_pass_gate(registry):
    registry[0].get_registered_model.return_value.aliases = {}
    decision = decide(registry)
    assert decision.promoted and decision.previous_version is None


def test_bootstrap_does_not_bypass_gate(registry):
    registry[0].get_registered_model.return_value.aliases = {}
    registry[2]["candidate"].data.metrics["test_accuracy"] = 0.2
    assert not decide(registry).promoted
    registry[0].set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("who", ["candidate", "champion"])
@pytest.mark.parametrize("fingerprint", [None, "", "different-split"])
def test_unknown_or_different_split_blocks_comparison(registry, who, fingerprint):
    registry[2][who] = make_run(0.36 if who == "candidate" else 0.34, fingerprint)
    assert not decide(registry).eligible
    registry[0].set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("who", ["candidate", "champion"])
def test_incomplete_evaluation_blocks_promotion(registry, who):
    registry[2][who].info.status = "FAILED"
    assert not decide(registry).promoted
    registry[0].set_registered_model_alias.assert_not_called()


def test_invalid_champion_metric_blocks_comparison(registry):
    registry[2]["champion"].data.metrics["test_macro_f1"] = float("nan")
    assert not decide(registry).promoted
    registry[0].set_registered_model_alias.assert_not_called()


def test_model_without_source_run_is_rejected(registry):
    registry[1]["2"].run_id = None
    assert not decide(registry).promoted
    registry[0].set_registered_model_alias.assert_not_called()


def test_model_not_ready_is_rejected(registry):
    registry[1]["2"].status = "PENDING_REGISTRATION"
    assert not decide(registry).promoted
    registry[0].set_registered_model_alias.assert_not_called()


def test_dry_run_is_read_only(registry):
    decision = decide(registry, dry_run=True)
    assert decision.eligible and not decision.promoted
    registry[0].set_registered_model_alias.assert_not_called()


def test_already_production_is_a_no_op(registry):
    registry[0].get_registered_model.return_value.aliases = {"production": "2"}
    assert not decide(registry).promoted
    registry[0].set_registered_model_alias.assert_not_called()


def test_concurrent_alias_change_blocks_write(registry):
    registry[0].get_registered_model.side_effect = [
        SimpleNamespace(aliases={"production": "1"}),
        SimpleNamespace(aliases={"production": "3"}),
    ]
    assert not decide(registry).promoted
    registry[0].set_registered_model_alias.assert_not_called()


def test_registry_errors_are_not_treated_as_empty_production(registry):
    registry[0].get_registered_model.side_effect = RuntimeError("Registry unavailable")
    with pytest.raises(RuntimeError, match="Registry unavailable"):
        decide(registry)
    registry[0].set_registered_model_alias.assert_not_called()


def test_cli_writes_json_decision_for_the_workflow(tmp_path, monkeypatch, capsys):
    module = importlib.import_module("trajectory_mlops.promote")
    decision = PromotionResult("2", "1", True, False, "Dry run.", "models:/classifier/2")
    mocked = Mock(return_value=decision)
    monkeypatch.setattr(module, "promote", mocked)
    path = tmp_path / "reports" / "promotion.json"
    module.main(["--version", "2", "--dry-run", "--output", str(path)])
    assert json.loads(path.read_text())["promoted"] is False
    assert json.loads(capsys.readouterr().out)["eligible"] is True
    mocked.assert_called_once_with("2", dry_run=True)
