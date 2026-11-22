# Backend and demo

The API loads one MLflow PyTorch model on CPU at startup. Configure MODEL_URI
with a numbered URI (recommended) or an alias; aliases resolve once to a fixed
number, returned in every prediction. Restart the API to change the served model.

Use the same MLflow registry as training and promotion. Initial model bootstrap
uses the committed Hoang handoff bundle; see [deployment](DEPLOYMENT.md).
No training runs are needed to serve that evaluated model.

```powershell
uv sync --frozen --extra ml --extra ui --extra dev
# Configure MLFLOW_TRACKING_URI and MODEL_URI in .env first.
uv run python -m uvicorn trajectory_mlops.api.main:app --host 127.0.0.1 --port 8000
# In another terminal:
uv run python -m streamlit run app/streamlit_app.py
```

| Route | Purpose |
| --- | --- |
| GET /health/live | Process liveness, including while no model is available |
| GET /health/ready | 200 when loaded, otherwise 503 |
| GET /model | Version, class order, preprocessing size and upload limits |
| POST /predict | Multipart field `file`, PNG or JPEG, five class scores |
| GET /docs | Interactive Swagger documentation |
| GET /openapi.json | Machine-readable API contract |

Images must come from the project's trajectory converter. A photograph or an
unrelated image is accepted as an image but its class scores are not meaningful.
Uploads default to 5 MiB and 4 million pixels, configurable through MAX_UPLOAD_BYTES
and MAX_IMAGE_PIXELS. Byte checks apply to the uploaded file; configure ingress
body limits if exposing the service publicly.

Class order and image size are checked against MLflow metadata before loading.
Output validation requires exactly five finite probabilities summing to one,
with confidence equal to the highest class score. Confidence is an uncalibrated
softmax score. The bundled model has limited minority-class performance; read
the handoff evaluation before interpreting a result.

Bad uploads return 413/415/422; missing model returns 503. Failed inference or an
invalid model response returns a controlled 500; details stay in service logs.

## Checks

```powershell
uv run python -m pytest -m "not e2e"
uv run python -m trajectory_mlops.api.smoke --url http://localhost:8000 --model-uri models:/trajectory-classifier/1
uv run python -m trajectory_mlops.api.smoke --url http://localhost:8000 --image data/raw/Safe/YOUR_WINDOW.png
```

The integration suite imports Hoang's bundled weights into a temporary SQLite
MLflow registry, loads them through application startup and performs HTTP inference.
It does not contact or modify the team's registry. The generated-image smoke
checks serving and version consistency, not accuracy. The deployed API test also
accepts E2E_API_URL and E2E_IMAGE_PATH to run against an authorized real window.
