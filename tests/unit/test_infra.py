import runpy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_compose_dependencies_and_services():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"minio", "init-bucket", "mlflow", "api", "ui"}
    for service in services.values():
        assert set(service.get("depends_on", {})) <= set(services)


def test_kubernetes_renderer_and_resources(monkeypatch):
    monkeypatch.chdir(ROOT)
    render = runpy.run_path("scripts/render_k8s.py")["render"]
    text = render("team", "abc123", "models:/trajectory-classifier/1")
    assert "__REGISTRY__" not in text and "__MODEL_URI_JSON__" not in text
    resources = list(yaml.safe_load_all(text))
    assert len(resources) == 6
    for resource in resources:
        assert resource["metadata"]["namespace"] == "trajectory-mlops"
    images = [
        r["spec"]["template"]["spec"]["containers"][0]["image"]
        for r in resources
        if r["kind"] == "Deployment"
    ]
    assert set(images) == {
        "team/trajectory-api:abc123",
        "team/trajectory-ui:abc123",
        "team/trajectory-tracking:abc123",
    }
    with pytest.raises(ValueError):
        render("team\nmalformed", "latest", "models:/a/1")
    with pytest.raises(ValueError):
        render("team", "bad tag", "models:/a/1")


def test_dag_has_expected_stages_and_real_dependencies():
    dag = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    assert list(dag["stages"]) == ["prepare", "train", "evaluate"]
    for stage in dag["stages"].values():
        for dep in stage["deps"]:
            if dep.startswith("src/") or dep in {"pyproject.toml", "uv.lock"}:
                assert (ROOT / dep).exists(), dep
