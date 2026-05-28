from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .base import BaseInference, Detection, InferenceResult
from ._draw import draw_detections


class FasterRCNNMobileNetInference(BaseInference):
    """torchvision FasterRCNN with MobileNetV3-Large backbone. Imports `create_model`
    from the original Faster-RCNN-Mobilnet repo so the model class does not drift."""

    name = "Faster-RCNN-MobileNet"
    default_threshold = 0.5

    def __init__(self, weight_path: str, device: str = "cuda", repo_path: str = ""):
        super().__init__(weight_path=weight_path, device=device)
        self.repo_path = repo_path

    def _load(self) -> None:
        repo = Path(self.repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"Faster-RCNN-Mobilnet repo not found at {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from model import create_model  # noqa

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.weight_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

        model = create_model(num_classes=2, pretrained=False)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        self.model = model
        self.torch_device = device

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        thr = threshold if threshold is not None else self.default_threshold
        img = image.convert("RGB")
        x = to_tensor(img).unsqueeze(0).to(self.torch_device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model(x)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        dets = [
            Detection(bbox=[float(x1), float(y1), float(x2), float(y2)], confidence=float(s))
            for (x1, y1, x2, y2), s in zip(boxes, scores)
            if s >= thr
        ]
        return InferenceResult(detections=dets, latency_ms=latency_ms,
                                annotated_image=draw_detections(img, dets))
