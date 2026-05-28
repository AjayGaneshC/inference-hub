from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from ._objbbox import _ObjBBoxBase, decode_yolo_norm


class _DenseNetDetector(nn.Module):
    """Matches densenet/densenet/densenet_training_v2.py:DenseNetDetector — the
    architecture the saved checkpoint at densenet_best.pt was trained with."""

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        base = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.objectness = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1), nn.Sigmoid(),
        )
        self.bbox = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 4), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.objectness(x), self.bbox(x)


class DenseNetInference(_ObjBBoxBase):
    name = "DenseNet"
    default_threshold = 0.5
    input_size = (128, 1024)
    # Training used only Resize + ToTensor (no Normalize).
    normalize_mean = None
    normalize_std = None
    bbox_decoder = staticmethod(decode_yolo_norm)
    objectness_already_sigmoid = True       # head includes nn.Sigmoid()

    def _load(self) -> None:
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        model = _DenseNetDetector().to(device)
        ckpt = torch.load(self.weight_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.model = model
        self.torch_device = device
        self._transform = self._build_transform()
