from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from ._objbbox import _ObjBBoxBase, decode_yolo_norm


class _ConvNeXtTinyDetector(nn.Module):
    """Mirrors convnext_tiny/convnext_training.py:convnexttiny exactly."""

    def __init__(self):
        super().__init__()
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.objectness = nn.Linear(768, 1)   # raw logits (BCEWithLogitsLoss)
        self.bbox = nn.Linear(768, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.objectness(x), self.bbox(x)


class ConvNeXtTinyInference(_ObjBBoxBase):
    name = "ConvNeXt-tiny"
    default_threshold = 0.5
    input_size = (512, 512)
    bbox_decoder = staticmethod(decode_yolo_norm)
    objectness_already_sigmoid = False        # raw logits → apply sigmoid

    def _load(self) -> None:
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        model = _ConvNeXtTinyDetector().to(device)
        ckpt = torch.load(self.weight_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # The training script handles a legacy 'objectness.0.weight' key form.
        remapped = {}
        for k, v in state_dict.items():
            remapped[k.replace("objectness.0.", "objectness.")] = v
        model.load_state_dict(remapped, strict=False)
        model.eval()
        self.model = model
        self.torch_device = device
        self._transform = self._build_transform()
