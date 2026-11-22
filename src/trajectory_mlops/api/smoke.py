"""HTTP smoke check, runnable inside the API image with no dev dependencies."""

import argparse
import json
import mimetypes
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

from trajectory_mlops.api.main import Prediction


def smoke(url: str, model_uri: str = "", image: Path | None = None) -> dict:
    url = url.rstrip("/")
    with urlopen(url + "/health/ready", timeout=60) as response:
        ready = json.load(response)
    if ready.get("status") != "ready":
        raise ValueError("API is not ready.")
    if image:
        content = image.read_bytes()
        mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    else:
        buffer = BytesIO()
        Image.new("RGB", (128, 128), "white").save(buffer, format="PNG")
        content, mime = buffer.getvalue(), "image/png"
    boundary = "trajectory-smoke-boundary"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="smoke.png"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        + content
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = Request(
        url + "/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urlopen(request, timeout=60) as response:
        result = Prediction.model_validate(json.load(response))
    if model_uri and result.model_version != model_uri.rsplit("/", 1)[-1]:
        raise ValueError("The API is serving a different model version.")
    return result.model_dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model-uri", default="")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    print(json.dumps(smoke(args.url, args.model_uri, args.image), indent=2))
