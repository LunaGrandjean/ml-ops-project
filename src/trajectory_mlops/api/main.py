"""FastAPI application: python -m uvicorn trajectory_mlops.api.main:app."""

import logging
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from trajectory_mlops.api.metrics import ApiMetrics, MetricsMiddleware
from trajectory_mlops.api.predictor import Predictor
from trajectory_mlops.api.schemas import Prediction
from trajectory_mlops.config import CLASSES, Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, predictor=None) -> FastAPI:
    settings = settings or Settings()
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor
        if predictor is None and settings.model_uri:
            try:
                app.state.predictor = await run_in_threadpool(Predictor, settings)
            except Exception:
                logger.exception("Model loading failed; readiness remains 503.")
        metrics.ready.set(int(app.state.predictor is not None))
        try:
            yield
        finally:
            metrics.ready.set(0)

    app = FastAPI(
        title="Trajectory Classifier",
        version="0.2.0",
        lifespan=lifespan,
        description=(
            "Classify a trajectory window as one of five driving categories. "
            "Upload the PNG/JPEG produced by the project converter. "
            "Confidence scores are softmax outputs and are not calibrated probabilities."
        ),
    )

    app.state.metrics = metrics
    app.add_middleware(MetricsMiddleware, metrics=metrics)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(content=metrics.render(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/health/live", tags=["health"], summary="Check that the process is alive")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"], summary="Check that a model is loaded")
    def ready(request: Request):
        if request.app.state.predictor is None:
            raise HTTPException(503, "Model unavailable. Configure MODEL_URI and restart the API.")
        return {"status": "ready"}

    @app.get("/model", tags=["inference"], summary="Inspect the loaded model and upload limits")
    def model_info(request: Request):
        engine = request.app.state.predictor
        if engine is None:
            raise HTTPException(503, "Model unavailable.")
        return {
            "model_version": engine.version,
            "model_uri": getattr(engine, "model_uri", ""),
            "image_size": getattr(engine, "image_size", None),
            "classes": list(CLASSES),
            "max_upload_bytes": settings.max_upload_bytes,
            "max_image_pixels": settings.max_image_pixels,
            "confidence_note": "Softmax scores are not calibrated probabilities.",
        }

    @app.post(
        "/predict",
        response_model=Prediction,
        tags=["inference"],
        summary="Classify one trajectory image",
        responses={
            413: {"description": "Upload exceeds the byte limit"},
            415: {"description": "Unsupported image format"},
            422: {"description": "Missing, corrupt or oversized image dimensions"},
            500: {"description": "Model inference failed"},
            503: {"description": "Model is unavailable"},
        },
    )
    async def predict(request: Request, file: Annotated[UploadFile, File()]):
        engine = request.app.state.predictor
        if engine is None:
            raise HTTPException(503, "Model unavailable.")
        if file.content_type not in {"image/png", "image/jpeg"}:
            raise HTTPException(415, "Upload a PNG or JPEG image.")
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "Image exceeds the upload limit.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        raise HTTPException(415, "Image content must be PNG or JPEG.")
                    if image.width * image.height > settings.max_image_pixels:
                        raise HTTPException(422, "Image dimensions exceed the pixel limit.")
                    image.load()
                    rgb = image.convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise HTTPException(422, "Invalid image or excessive image dimensions.") from exc
        try:
            result = await run_in_threadpool(engine.predict, rgb)
            validated = Prediction.model_validate(result)
            metrics.prediction(validated.label, validated.confidence)
            return validated
        except Exception as exc:
            logger.exception("Model inference failed.")
            raise HTTPException(500, "Prediction failed. Check the API service logs.") from exc

    return app


app = create_app()
