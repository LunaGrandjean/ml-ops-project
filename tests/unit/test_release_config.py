import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release_config", Path(__file__).parents[2] / "scripts/release_config.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("uri", ["models:/trajectory-classifier/2", ""])
def test_valid_release_inputs(uri):
    assert module.release_config("a" * 40, uri, False)["model_uri"] == uri


@pytest.mark.parametrize(
    "uri", ["", "models:/name@production", "models:/name/0", "models:/name/1\nfoo=bar"]
)
def test_deployment_rejects_mutable_or_invalid_uri(uri):
    with pytest.raises(ValueError):
        module.release_config("a" * 40, uri, True)


def test_release_rejects_unresolved_ref():
    with pytest.raises(ValueError):
        module.release_config("main", "models:/name/1", True)
