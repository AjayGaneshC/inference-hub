from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from .base import BaseInference, Detection, InferenceResult


# Path inside the container where the original VIT-Artery repo is mounted.
# Mirrors host: ~/Ajay-Codes/Models-trained-on-Artery-data/VIT-Artery
VIT_REPO_DEFAULT = "/workspace/refs/VIT-Artery"


class ViTArteryInference(BaseInference):
    """ViT detector — imports the original `ViTArteryDetector` class from VIT-Artery/inference.py
    rather than re-declaring it, so the wrapper does not drift from training code."""

    name = "ViT-Artery"
    default_threshold = 0.5

    def __init__(self, weight_path: str, device: str = "cuda", repo_path: str = VIT_REPO_DEFAULT):
        super().__init__(weight_path=weight_path, device=device)
        self.repo_path = repo_path
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load(self) -> None:
        repo = Path(self.repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"VIT-Artery repo not found at {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from inference import ViTArteryDetector  # noqa: E402

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.weight_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        model = ViTArteryDetector(num_classes=1, max_objects=5, pretrained=False).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.torch_device = device

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        thr = threshold if threshold is not None else self.default_threshold
        img = image.convert("RGB")
        W, H = img.size
        x = self._transform(img).unsqueeze(0).to(self.torch_device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        bbox = out["bbox"][0]                          # [max_obj, 4] (already sigmoid'd in forward)
        objn = torch.sigmoid(out["objectness"][0])      # [max_obj]

        dets: list[Detection] = []
        for i in range(bbox.size(0)):
            c = float(objn[i].item())
            if c < thr:
                continue
            cx, cy, bw, bh = bbox[i].tolist()
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            dets.append(Detection(bbox=[x1, y1, x2, y2], confidence=c))

        annotated = self._draw(img.copy(), dets)
        return InferenceResult(detections=dets, latency_ms=latency_ms, annotated_image=annotated)

    @staticmethod
    def _draw(img: Image.Image, dets: list[Detection]) -> Image.Image:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        for d in dets:
            draw.rectangle(d.bbox, outline="red", width=3)
            label = f"{d.label}: {d.confidence:.2f}"
            tx, ty = d.bbox[0], max(0, d.bbox[1] - 20)
            draw.text((tx, ty), label, fill="red", font=font)
        return img
