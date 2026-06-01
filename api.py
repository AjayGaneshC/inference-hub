"""FastAPI inference hub. Replaces the Gradio UI with a JSON API consumed by the
Next.js frontend in web/. Models are lazy-loaded on first request and LRU-cached."""

from __future__ import annotations

import base64
import io
import os
import traceback
from dataclasses import asdict
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from registry import MODEL_REGISTRY


@lru_cache(maxsize=None)
def get_model(name: str):
    inst = MODEL_REGISTRY[name]()
    inst.load()
    return inst


_PREVIEW_MAX_SIDE = int(os.environ.get("PREVIEW_MAX_SIDE", "1024"))
_PREVIEW_JPEG_QUALITY = int(os.environ.get("PREVIEW_JPEG_QUALITY", "82"))


def _encode_preview_b64(img: Image.Image) -> str:
    rgb = img.convert("RGB")
    w, h = rgb.size
    m = max(w, h)
    if m > _PREVIEW_MAX_SIDE:
        s = _PREVIEW_MAX_SIDE / m
        rgb = rgb.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=_PREVIEW_JPEG_QUALITY, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class DetectionOut(BaseModel):
    bbox: List[float]
    confidence: float
    label: str


class ModelResult(BaseModel):
    model: str
    latency_ms: float
    detections: List[DetectionOut]
    annotated_image_b64: Optional[str] = None
    error: Optional[str] = None


class InferResponse(BaseModel):
    results: List[ModelResult]


app = FastAPI(title="Vessel Vision Inference Hub API")

# CORS: localhost for dev, configurable extra origins for the Vercel deploy.
_extra = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_extra,
    allow_origin_regex=r"^(https?://localhost(:\d+)?|https?://127\.0\.0\.1(:\d+)?|https://.*\.vercel\.app)$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models")
def list_models():
    return {"models": sorted(MODEL_REGISTRY.keys())}


@app.post("/infer", response_model=InferResponse)
async def infer(
    image: UploadFile = File(...),
    models: str = Form(...),  # comma-separated
    threshold: float = Form(0.5),
):
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    names = [n.strip() for n in models.split(",") if n.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="No models selected")

    unknown = [n for n in names if n not in MODEL_REGISTRY]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown models: {unknown}")

    results: List[ModelResult] = []
    for name in names:
        try:
            model = get_model(name)
            r = model.predict(pil, threshold=threshold)
            annotated = r.annotated_image if r.annotated_image is not None else pil
            results.append(
                ModelResult(
                    model=name,
                    latency_ms=r.latency_ms,
                    detections=[DetectionOut(**asdict(d)) for d in r.detections],
                    annotated_image_b64=_encode_preview_b64(annotated),
                )
            )
        except Exception as e:
            traceback.print_exc()
            results.append(
                ModelResult(model=name, latency_ms=0.0, detections=[], error=str(e))
            )

    return InferResponse(results=results)
