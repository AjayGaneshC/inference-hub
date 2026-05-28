from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .base import BaseInference, Detection, InferenceResult
from ._draw import draw_detections


class RFDETRInference(BaseInference):
    """RF-DETR using the in-repo `rfdetr` package shipped with the model directory.
    Loads via `RFDETRBase(pretrain_weights=<path>, num_classes=1)` and uses its
    high-level `.predict(image, threshold)` API which returns supervision.Detections."""

    name = "RF-DETR"
    default_threshold = 0.5

    def __init__(self, weight_path: str, device: str = "cuda",
                 repo_path: str = "", resolution: int = 644, num_classes: int = 1):
        super().__init__(weight_path=weight_path, device=device)
        self.repo_path = repo_path
        self.resolution = resolution
        self.num_classes = num_classes

    def _load(self) -> None:
        repo = Path(self.repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"RF-DETR repo not found at {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from rfdetr import RFDETRBase  # noqa

        self.model = RFDETRBase(
            pretrain_weights=self.weight_path,
            num_classes=self.num_classes,
            resolution=self.resolution,
        )

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        thr = threshold if threshold is not None else self.default_threshold

        t0 = time.perf_counter()
        sv_det = self.model.predict(image.convert("RGB"), threshold=thr)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        dets: list[Detection] = []
        xyxy = getattr(sv_det, "xyxy", None)
        conf = getattr(sv_det, "confidence", None)
        if xyxy is not None and len(xyxy) > 0:
            xyxy = np.asarray(xyxy)
            conf = np.asarray(conf) if conf is not None else np.ones(len(xyxy))
            for (x1, y1, x2, y2), s in zip(xyxy, conf):
                dets.append(Detection(bbox=[float(x1), float(y1), float(x2), float(y2)],
                                      confidence=float(s)))

        return InferenceResult(detections=dets, latency_ms=latency_ms,
                                annotated_image=draw_detections(image, dets))
