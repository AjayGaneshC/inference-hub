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
    # The artery-trained RT-DETR is the run5 checkpoint (RTDETRDetectionModel,
    # classes ["Background", "Artery"]) loaded by the repo's own test.py/train.py.
    # NOTE: rtdetr-l.pt / yolo11n.pt in this folder are the *stock COCO* downloads,
    # NOT artery-trained — do not point production at them.
    "RT-DETR-L": lambda: RTDETRInference(
        weight_path=_w("Models-trained-on-Artery-data/RTDETR/runs/detect/train5/weights/best.pt"),
        device=DEVICE,
    ),
    # Artery-trained YOLO11n (task=detect, classes ["Artery"]), loaded via the
    # ultralytics YOLO class like RT-DETR. The checkpoint ships in the repo root,
    # which is mounted into the container under /weights/inference-hub.
    "YOLO 11n": lambda: RTDETRInference(
        weight_path=_w("inference-hub/11n_new_best.pt"),
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
        weight_path=_w("Models-trained-on-Artery-data/RF-DETR/output_artery/checkpoint.pth"),
        device=DEVICE,
        repo_path=_r("RF-DETR/rf-detr"),
        # RF-DETR's windowed attention requires resolution divisible by 14*4=56.
        # The checkpoint's saved arg is 644 (not a multiple of 56), which crashes
        # at inference ("shape '[1,4,11,4,11,-1]' is invalid"): a 644px square is a
        # 46x46 patch grid, but the windowing targets 44x44. 616 (=11*56) is the
        # valid resolution the model effectively rounds 644 down to.
        resolution=616,
        num_classes=1,
        out_feature_indexes=[9],
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
