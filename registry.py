"""Maps display name → factory that builds a BaseInference. Lazy: factory is only called
when the model is first selected. Weight paths are resolved relative to WEIGHTS_ROOT,
which inside Docker is /weights (bind-mounted from ~/Ajay-Codes on the host).
REFS_ROOT points at /workspace/refs, where the original training repos are mounted so
their model classes can be imported via sys.path."""

from __future__ import annotations

import os
from typing import Callable, Dict

from models.base import BaseInference
from models.rtdetr import RTDETRInference
from models.vit import ViTArteryInference
from models.faster_rcnn import FasterRCNNMobileNetInference
from models.mobiledet import MobileDetSSDLiteInference
from models.nanodet import NanoDetInference
from models.rfdetr import RFDETRInference
from models.densenet import DenseNetInference
from models.convnext import ConvNeXtTinyInference
from models.squeezenet import SqueezeNetInference
from models.xception import XceptionInference

WEIGHTS_ROOT = os.environ.get("WEIGHTS_ROOT", "/weights")
DOWNLOADS_ROOT = os.environ.get("DOWNLOADS_ROOT", "/downloads")
REFS_ROOT = os.environ.get("REFS_ROOT", "/workspace/refs")
DEVICE = os.environ.get("INFERENCE_DEVICE", "cuda")


def _w(*parts: str) -> str:
    return os.path.join(WEIGHTS_ROOT, *parts)


def _d(*parts: str) -> str:
    """For weights that live under the user's ~/Downloads folder."""
    return os.path.join(DOWNLOADS_ROOT, *parts)


def _r(*parts: str) -> str:
    return os.path.join(REFS_ROOT, *parts)


# name → factory. Models are loaded lazily on first selection.
MODEL_REGISTRY: Dict[str, Callable[[], BaseInference]] = {
    "RT-DETR-L": lambda: RTDETRInference(
        weight_path=_w("Models-trained-on-Artery-data/RTDETR/rtdetr-l.pt"),
        device=DEVICE,
    ),
    "YOLO 11n": lambda: RTDETRInference(
        weight_path=_w("Models-trained-on-Artery-data/RTDETR/yolo11n.pt"),
        device=DEVICE,
    ),
    "ViT-Artery": lambda: ViTArteryInference(
        weight_path=_w("Models-trained-on-Artery-data/VIT-Artery/VIT.pth"),
        device=DEVICE,
        repo_path=_r("VIT-Artery"),
    ),
    "Faster-RCNN-MobileNet": lambda: FasterRCNNMobileNetInference(
        weight_path=_w("Models-trained-on-Artery-data/Faster-RCNN-Mobilnet/outputs/best_model.pth"),
        device=DEVICE,
        repo_path=_r("Faster-RCNN-Mobilnet"),
    ),
    "MobileDet-SSDLite": lambda: MobileDetSSDLiteInference(
        weight_path=_w("MobileDet/outputs/best_hyperparam/best_model.pth"),
        device=DEVICE,
        repo_path=_r("MobileDet"),
    ),
    "NanoDet": lambda: NanoDetInference(
        weight_path=_w("Models-trained-on-Artery-data/NanoDet/workspace/nanodet_custom/model_epoch_19.pth"),
        device=DEVICE,
        repo_path=_r("NanoDet"),
        config_path=_r("NanoDet/config/nanodet_custom.yml"),
    ),
    "RF-DETR": lambda: RFDETRInference(
        weight_path=_w("Models-trained-on-Artery-data/RF-DETR/rf-detr-base.pth"),
        device=DEVICE,
        repo_path=_r("RF-DETR/rf-detr"),
        resolution=644,
        num_classes=1,
    ),
    "DenseNet": lambda: DenseNetInference(
        weight_path=_w("densenet/checkpoints/densenet_best.pt"),
        device=DEVICE,
    ),
    "ConvNeXt-tiny": lambda: ConvNeXtTinyInference(
        weight_path=_d("convnext_tiny-20251119T052525Z-1-001/convnext_tiny/checkpoints/convnext_tiny_best.pt"),
        device=DEVICE,
    ),
    "SqueezeNet": lambda: SqueezeNetInference(
        weight_path=_w("squeeze-inference/squeezenet_optimized_best.pth"),
        device=DEVICE,
    ),
    "Xception": lambda: XceptionInference(
        weight_path=_d("Xception-codes-20251029T053922Z-1-001/Xception-codes/xception_detection_best.pth"),
        device=DEVICE,
        repo_path=_r("Xception-codes"),
    ),
}
