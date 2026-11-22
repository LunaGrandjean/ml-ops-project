"""Resolve immutable release inputs without interpolating them into shell code."""

import os
import re
import subprocess
from pathlib import Path


def release_config(sha: str, model_uri: str, deploying: bool) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Release source must resolve to a full Git commit SHA.")
    if (deploying or model_uri) and not re.fullmatch(r"models:/[\w.-]+/[1-9][0-9]*", model_uri):
        raise ValueError("Deployment requires a numbered models:/name/version URI.")
    return {"sha": sha, "model_uri": model_uri}


if __name__ == "__main__":
    config = release_config(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        os.getenv("RELEASE_MODEL_URI", ""),
        os.getenv("DEPLOY_ENABLED") == "true",
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        for key, value in config.items():
            stream.write(f"{key}={value}\n")
