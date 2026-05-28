from __future__ import annotations

import time

import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms

from .base import BaseInference, Detection, InferenceResult
from ._draw import draw_detections


class _SqueezeNetWithBBox(nn.Module):
    """Mirrors densenet/Squeezenet/squeezenet_training_optimized.py exactly."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.features = tvm.squeezenet1_1(weights=tvm.SqueezeNet1_1_Weights.DEFAULT).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.bbox_regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(512, 4, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        cls = self.avgpool(self.classifier(x)).view(x.size(0), -1)
        bbox = self.avgpool(self.bbox_regressor(x)).view(x.size(0), -1)
        return cls, bbox


class SqueezeNetInference(BaseInference):
    """SqueezeNet with two heads (binary class softmax + bbox xyxy normalized 0-1)."""

    name = "SqueezeNet"
    default_threshold = 0.5

    def _load(self) -> None:
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        model = _SqueezeNetWithBBox(num_classes=2).to(device)
        ckpt = torch.load(self.weight_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.model = model
        self.torch_device = device
        self._transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        thr = threshold if threshold is not None else self.default_threshold
        img = image.convert("RGB")
        W, H = img.size
        x = self._transform(img).unsqueeze(0).to(self.torch_device)

        t0 = time.perf_counter()
        with torch.no_grad():
            cls, bbox = self.model(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Class logits (after ReLU): treat softmax[1] as artery probability.
        probs = torch.softmax(cls, dim=1)
        score = float(probs[0, 1].item())

        dets: list[Detection] = []
        if score >= thr:
            x1n, y1n, x2n, y2n = bbox.flatten().tolist()[:4]
            dets.append(Detection(
                bbox=[x1n * W, y1n * H, x2n * W, y2n * H],
                confidence=score,
            ))

        return InferenceResult(detections=dets, latency_ms=latency_ms,
                                annotated_image=draw_detections(img, dets))
