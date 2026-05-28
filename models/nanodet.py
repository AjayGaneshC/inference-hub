from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .base import BaseInference, Detection, InferenceResult
from ._draw import draw_detections


class NanoDetInference(BaseInference):
    """NanoDet — mirrors NanoDet/inference.py. Uses BGR cv2 input with the
    specific (104, 117, 124) / (57.4, 57.1, 58.4) normalization the nanodet
    repo expects, then runs the model's own `head.post_process`."""

    name = "NanoDet"
    default_threshold = 0.35

    _MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
    _STD = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, weight_path: str, device: str = "cuda",
                 repo_path: str = "", config_path: str = ""):
        super().__init__(weight_path=weight_path, device=device)
        self.repo_path = repo_path
        self.config_path = config_path

    def _load(self) -> None:
        repo = Path(self.repo_path)
        cfg_path = Path(self.config_path)
        if not repo.exists():
            raise FileNotFoundError(f"NanoDet repo not found at {repo}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"NanoDet config not found at {cfg_path}")

        # The original inference.py adds the inner `nanodet/` dir to sys.path.
        for p in (str(repo), str(repo / "nanodet")):
            if p not in sys.path:
                sys.path.insert(0, p)

        from nanodet.util import cfg, load_config  # noqa
        from nanodet.model.arch import build_model  # noqa

        load_config(cfg, str(cfg_path))
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        model = build_model(cfg.model)
        ckpt = torch.load(self.weight_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()

        self.model = model
        self.cfg = cfg
        self.torch_device = device
        self.class_names = getattr(cfg, "class_names", ["artery"])

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        thr = threshold if threshold is not None else self.default_threshold

        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        norm = (bgr - self._MEAN) / self._STD
        chw = norm.transpose(2, 0, 1)
        x = torch.from_numpy(chw).unsqueeze(0).to(self.torch_device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(x)
            results = self.model.head.post_process(outputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        arr = results[0].cpu().numpy() if hasattr(results[0], "cpu") else np.asarray(results[0])
        dets: list[Detection] = []
        for row in arr:
            x1, y1, x2, y2, score, cls_id = row[:6]
            if score < thr:
                continue
            label = self.class_names[int(cls_id)] if 0 <= int(cls_id) < len(self.class_names) else "obj"
            dets.append(Detection(bbox=[float(x1), float(y1), float(x2), float(y2)],
                                  confidence=float(score), label=label))

        return InferenceResult(detections=dets, latency_ms=latency_ms,
                                annotated_image=draw_detections(image, dets))
