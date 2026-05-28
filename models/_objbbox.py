"""Shared helper for the "single-object detector" family — backbones with one
objectness head + one bbox head (DenseNet, ConvNeXt-tiny, SqueezeNet, Xception).
Each subclass plugs in its own model class, input size, normalize stats, and
bbox decoding (YOLO-format vs xyxy vs cxcywh)."""

from __future__ import annotations

import time
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms

from .base import BaseInference, Detection, InferenceResult
from ._draw import draw_detections


def decode_yolo_norm(bbox: list[float], W: int, H: int) -> list[float]:
    """[x_c, y_c, w, h] normalized → [x1, y1, x2, y2] in pixels."""
    cx, cy, bw, bh = bbox
    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H
    return [x1, y1, x2, y2]


def decode_xyxy_norm(bbox: list[float], W: int, H: int) -> list[float]:
    """[x1, y1, x2, y2] normalized → pixels."""
    x1, y1, x2, y2 = bbox
    return [x1 * W, y1 * H, x2 * W, y2 * H]


class _ObjBBoxBase(BaseInference):
    """Common predict() for object-or-not + single bbox models."""

    input_size: tuple[int, int] = (224, 224)
    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]
    channels: int = 3                       # 1 for grayscale (Xception)
    bbox_decoder: Callable[[list[float], int, int], list[float]] = staticmethod(decode_yolo_norm)
    # Whether the model's objectness head already includes sigmoid.
    objectness_already_sigmoid: bool = False

    def _build_transform(self) -> transforms.Compose:
        steps: list = [transforms.Resize(self.input_size)]
        if self.channels == 1:
            steps.append(transforms.Grayscale(num_output_channels=1))
        steps.append(transforms.ToTensor())
        if self.normalize_mean is not None:
            steps.append(transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std))
        return transforms.Compose(steps)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Subclasses override only if the model's forward order isn't (obj, bbox)."""
        out = self.model(x)
        # Accept both (obj, bbox) and (bbox, obj) orderings.
        return out

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        thr = threshold if threshold is not None else self.default_threshold
        img = image.convert("RGB")
        W, H = img.size
        x = self._transform(img).unsqueeze(0).to(self.torch_device)

        t0 = time.perf_counter()
        with torch.no_grad():
            obj, bbox = self._forward(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        score = obj.flatten()[0].item()
        if not self.objectness_already_sigmoid:
            score = float(torch.sigmoid(torch.tensor(score)).item())

        dets: list[Detection] = []
        if score >= thr:
            box_norm = bbox.flatten().tolist()[:4]
            dets.append(Detection(bbox=self.bbox_decoder(box_norm, W, H), confidence=score))

        return InferenceResult(detections=dets, latency_ms=latency_ms,
                                annotated_image=draw_detections(img, dets),
                                extra={"raw_score": score})
