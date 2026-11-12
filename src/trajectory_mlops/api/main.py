"""uv run python -m uvicorn trajectory_mlops.api.main:app --reload"""

import logging
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from trajectory_mlops.api.predictor import Predictor
from trajectory_mlops.config import Settings

logger = logging.getLogger(__name__)


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str


def create_app(settings: Settings | None = None, predictor=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor
        if predictor is None and settings.model_uri:
            try:
                app.state.predictor = await run_in_threadpool(Predictor, settings)
            except Exception:
                logger.exception("Chargement du modèle impossible ; readiness reste à 503.")
        yield

    app = FastAPI(title="Trajectory Classifier", version="0.1.0", lifespan=lifespan)

    @app.get("/health/live", tags=["health"])
    def live():
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request):
        if request.app.state.predictor is None:
            raise HTTPException(503, "Modèle non chargé : entraîner puis configurer MODEL_URI.")
        return {"status": "ready"}

    @app.post("/predict", response_model=Prediction, tags=["inference"])
    async def predict(request: Request, file: Annotated[UploadFile, File()]):
        engine = request.app.state.predictor
        if engine is None:
            raise HTTPException(503, "Modèle non chargé.")
        if file.content_type not in {"image/png", "image/jpeg"}:
            raise HTTPException(415, "Envoyer une image PNG ou JPEG.")
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "Image trop volumineuse.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        raise HTTPException(415, "Le contenu doit être une image PNG ou JPEG.")
                    image.load()
                    rgb = image.convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise HTTPException(422, "Image invalide ou dimensions excessives.") from exc
        return await run_in_threadpool(engine.predict, rgb)

    return app


app = create_app()
