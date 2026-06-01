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
    """NanoDet — mirrors the official nanodet/demo/demo.py: builds meta via the
    repo's Pipeline (which fills in warp_matrix etc.) and calls model.inference(meta),
    which internally invokes head.post_process(preds, meta)."""

    name = "NanoDet"
    default_threshold = 0.35

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

        for p in (str(repo), str(repo / "nanodet")):
            if p not in sys.path:
                sys.path.insert(0, p)

        from nanodet.util import cfg, load_config  # noqa
        from nanodet.model.arch import build_model  # noqa
        from nanodet.data.transform import Pipeline  # noqa

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
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.class_names = getattr(cfg, "class_names", ["artery"])

    def predict(self, image: Image.Image, threshold=None) -> InferenceResult:
        self.load()
        thr = threshold if threshold is not None else self.default_threshold

        from nanodet.data.batch_process import stack_batch_img  # noqa
        from nanodet.data.collate import naive_collate  # noqa

        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        img_info = {"id": 0, "file_name": None, "height": h, "width": w}
        meta = dict(img_info=img_info, raw_img=bgr, img=bgr)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.torch_device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        t0 = time.perf_counter()
        with torch.no_grad():
            results = self.model.inference(meta)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # results: { batch_idx: { class_idx: [[x1, y1, x2, y2, score], ...] } }
        dets: list[Detection] = []
        per_image = results[0] if isinstance(results, dict) else results
        for cls_id, boxes in per_image.items():
            label = self.class_names[int(cls_id)] if 0 <= int(cls_id) < len(self.class_names) else "obj"
            for row in boxes:
                x1, y1, x2, y2, score = row[:5]
                if score < thr:
                    continue
                dets.append(Detection(bbox=[float(x1), float(y1), float(x2), float(y2)],
                                      confidence=float(score), label=label))

        return InferenceResult(detections=dets, latency_ms=latency_ms,
                                annotated_image=draw_detections(image, dets))
