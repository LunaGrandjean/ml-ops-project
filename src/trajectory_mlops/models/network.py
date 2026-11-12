"""Petit CNN sans téléchargement de poids : baseline deep learning à améliorer."""

import torch
from torch import nn

from trajectory_mlops.config import CLASSES


class TrajectoryCNN(nn.Module):
    def __init__(self, image_size: int = 128):
        super().__init__()
        self.image_size = image_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, len(CLASSES))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images).flatten(1))
