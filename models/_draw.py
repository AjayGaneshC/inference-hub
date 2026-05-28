"""Shared helpers: drawing boxes and confidence labels onto a PIL image."""

from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .base import Detection


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_detections(img: Image.Image, dets: Sequence[Detection], color: str = "red") -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(16)
    for d in dets:
        draw.rectangle(d.bbox, outline=color, width=3)
        text = f"{d.label}: {d.confidence:.2f}"
        tx, ty = d.bbox[0], max(0, d.bbox[1] - 20)
        draw.text((tx, ty), text, fill=color, font=font)
    return out
