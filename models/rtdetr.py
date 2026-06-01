from __future__ import annotations

import time
import numpy as np
from PIL import Image

from .base import BaseInference, Detection, InferenceResult


class RTDETRInference(BaseInference):
    """RT-DETR-L trained via ultralytics (loaded with the YOLO class — matches RTDETR/test.py)."""

    name = "RT-DETR-L"
    default_threshold = 0.7

    def _load(self) -> None:
        from ultralytics import YOLO
        self.model = YOLO(self.weight_path)
        self.model.to(self.device)

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        conf = threshold if threshold is not None else self.default_threshold
        img_rgb = image.convert("RGB")

        # Pass the PIL image, not a bare numpy array: ultralytics' RT-DETR
        # postprocess (8.4.x) mishandles a single ndarray source and raises
        # "'list' object has no attribute 'shape'". A PIL image (like the file
        # paths the original RTDETR/test.py uses) avoids that code path.
        t0 = time.perf_counter()
        result = self.model.predict(source=img_rgb, conf=conf, iou=0.7, save=False, verbose=False)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        dets: list[Detection] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                dets.append(Detection(bbox=[float(x1), float(y1), float(x2), float(y2)],
                                      confidence=float(s)))

        # Ultralytics .plot() returns BGR numpy array
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated = Image.fromarray(annotated_rgb)

        return InferenceResult(detections=dets, latency_ms=latency_ms, annotated_image=annotated)
