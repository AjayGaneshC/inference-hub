from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image


@dataclass
class Detection:
    bbox: List[float]            # [x1, y1, x2, y2] in original image pixels
    confidence: float
    label: str = "vessel"


@dataclass
class InferenceResult:
    detections: List[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    annotated_image: Optional[Image.Image] = None
    extra: dict = field(default_factory=dict)


class BaseInference:
    """Each model wrapper subclasses this and implements _load and predict."""

    name: str = "base"
    default_threshold: float = 0.5

    def __init__(self, weight_path: str, device: str = "cuda"):
        self.weight_path = weight_path
        self.device = device
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self) -> None:
        raise NotImplementedError

    def predict(self, image: Image.Image, threshold: Optional[float] = None) -> InferenceResult:
        raise NotImplementedError
