from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .base import BaseInference, Detection, InferenceResult
from ._draw import draw_detections


class MobileDetSSDLiteInference(BaseInference):
    """MobileDet SSDLite with MobileNetV3-Large backbone. Imports
    `create_mobilenet_ssdlite_model` from the MobileDet repo."""

    name = "MobileDet-SSDLite"
    default_threshold = 0.5

    def __init__(self, weight_path: str, device: str = "cuda", repo_path: str = ""):
        super().__init__(weight_path=weight_path, device=device)
        self.repo_path = repo_path

    def _load(self) -> None:
        repo = Path(self.repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"MobileDet repo not found at {repo}")
        # The MobileDet repo's `models/mobilenet_ssdlite.py` collides with our own
        # top-level `models/` package, so load it by file path under a unique name.
        ssd_path = repo / "models" / "mobilenet_ssdlite.py"
        if not ssd_path.exists():
            raise FileNotFoundError(f"mobilenet_ssdlite.py not found at {ssd_path}")
        spec = importlib.util.spec_from_file_location("_mobiledet_ssdlite", ssd_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        create_mobilenet_ssdlite_model = mod.create_mobilenet_ssdlite_model

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.weight_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        else:
            state_dict = ckpt

        model = create_mobilenet_ssdlite_model(num_classes=2, backbone_name="mobilenet_v3_large")
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
            out = self.model(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Output may be list[dict] (torchvision style) or dict.
        pred = out[0] if isinstance(out, list) else out
        boxes = pred["boxes"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        dets = [
            Detection(bbox=[float(x1), float(y1), float(x2), float(y2)], confidence=float(s))
            for (x1, y1, x2, y2), s in zip(boxes, scores)
            if s >= thr
        ]
        return InferenceResult(detections=dets, latency_ms=latency_ms,
                                annotated_image=draw_detections(img, dets))
