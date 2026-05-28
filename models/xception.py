from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .base import BaseInference, Detection, InferenceResult
from ._draw import draw_detections


class XceptionInference(BaseInference):
    """Xception detector — imports `XceptionArteryDetector(mode='detection')` from the
    original Xception-codes repo. Grayscale 299x299 input, output (bbox, confidence)."""

    name = "Xception"
    default_threshold = 0.5

    def __init__(self, weight_path: str, device: str = "cuda", repo_path: str = ""):
        super().__init__(weight_path=weight_path, device=device)
        self.repo_path = repo_path

    def _load(self) -> None:
        repo = Path(self.repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"Xception-codes repo not found at {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from xception_model import XceptionArteryDetector  # noqa

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        model = XceptionArteryDetector(mode="detection", pretrained=False).to(device)
        ckpt = torch.load(self.weight_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.model = model
        self.torch_device = device
        self._transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        thr = threshold if threshold is not None else self.default_threshold
        img_rgb = image.convert("RGB")
        W, H = img_rgb.size
        x = self._transform(img_rgb).unsqueeze(0).to(self.torch_device)

        t0 = time.perf_counter()
        with torch.no_grad():
            bbox, confidence = self.model(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # bbox is already sigmoid'd in forward(); confidence head is raw logits.
        score = float(torch.sigmoid(confidence.flatten()[0]).item())

        dets: list[Detection] = []
        if score >= thr:
            cx, cy, bw, bh = bbox.flatten().tolist()[:4]
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            dets.append(Detection(bbox=[x1, y1, x2, y2], confidence=score))

        return InferenceResult(detections=dets, latency_ms=latency_ms,
                                annotated_image=draw_detections(img_rgb, dets))
