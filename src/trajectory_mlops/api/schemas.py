"""Shared HTTP response contract, also used by UI and release smoke checks."""

from math import isclose
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trajectory_mlops.config import CLASSES

ClassLabel = Literal["Safe", "Safe Overall", "Bad Weather", "Tandem", "Reckless"]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: ClassLabel
    confidence: Score
    probabilities: dict[str, Score]
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_scores(self) -> Self:
        if set(self.probabilities) != set(CLASSES):
            raise ValueError("All five class probabilities are required.")
        if not isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one.")
        selected = self.probabilities[self.label]
        if not isclose(self.confidence, selected, abs_tol=1e-6):
            raise ValueError("Confidence must match the predicted class score.")
        if selected < max(self.probabilities.values()):
            raise ValueError("Label must have the highest score.")
        return self
