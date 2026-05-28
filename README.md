# Vessel Vision Inference Hub

Local-GPU inference service for 11 benchmarked vessel-detection models, with a
Next.js frontend for side-by-side comparison.

- **Backend** (`api.py`) — FastAPI on port 8000, runs in Docker against the host GPU.
- **Frontend** (`web/`) — Next.js app, deployed to Vercel. Talks to the local backend.

## Models

| Model | Backend | Weight |
| --- | --- | --- |
| RT-DETR-L | ultralytics | `Models-trained-on-Artery-data/RTDETR/rtdetr-l.pt` |
| YOLO 11n | ultralytics | `Models-trained-on-Artery-data/RTDETR/yolo11n.pt` |
| ViT-Artery | custom torch | `Models-trained-on-Artery-data/VIT-Artery/VIT.pth` |
| Faster-RCNN-MobileNet | torchvision detection | `Models-trained-on-Artery-data/Faster-RCNN-Mobilnet/outputs/best_model.pth` |
| MobileDet-SSDLite | torchvision detection | `MobileDet/outputs/best_hyperparam/best_model.pth` |
| NanoDet | nanodet repo | `Models-trained-on-Artery-data/NanoDet/workspace/nanodet_custom/model_epoch_19.pth` |
| RF-DETR | rfdetr lib | `Models-trained-on-Artery-data/RF-DETR/rf-detr-base.pth` |
| DenseNet | custom torch | `densenet/checkpoints/densenet_best.pt` |
| ConvNeXt-tiny | custom torch | `Downloads/.../convnext_tiny_best.pt` |
| SqueezeNet | custom torch | `squeeze-inference/squeezenet_optimized_best.pth` |
| Xception | custom torch | `Downloads/.../xception_detection_best.pth` |

(YOLO 11n shares the ultralytics backend with RT-DETR — 11 wrappers cover 12 benchmark entries.)

For each model, the wrapper imports the **original training/inference code** off the
host via `sys.path.insert` rather than re-declaring the `nn.Module` class. This way
the model definition cannot drift from training. The repos are bind-mounted read-only
into `/workspace/refs/<name>` inside the container.

## Layout

```
inference-hub/
├── api.py              FastAPI backend (GET /models, POST /infer)
├── web/                Next.js frontend (image upload + model multi-select + gallery)
├── registry.py         name → lazy factory (LRU-cached on first use)
├── models/
│   ├── base.py         BaseInference, Detection, InferenceResult
│   ├── _draw.py        Shared box-drawing helpers
│   ├── _objbbox.py     Shared base for (objectness + bbox) single-detection models
│   ├── rtdetr.py       ultralytics-backed (RT-DETR-L + YOLO 11n)
│   ├── vit.py
│   ├── faster_rcnn.py
│   ├── mobiledet.py
│   ├── nanodet.py
│   ├── rfdetr.py
│   ├── densenet.py
│   ├── convnext.py
│   ├── squeezenet.py
│   └── xception.py
├── requirements.txt
├── Dockerfile          nvidia/cuda:12.1.1 + torch 2.4.1+cu121 + torchvision 0.19.1+cu121
└── docker-compose.yml  mounts ~/Ajay-Codes → /weights and ~/Downloads → /downloads
```

## Run

Backend:

```bash
cd ~/Ajay-Codes/inference-hub
docker compose build
docker compose up
# API at http://localhost:8000  (try GET /models)
```

Frontend (dev):

```bash
cd web
npm install
npm run dev
# open http://localhost:3000
```

For the deployed Vercel frontend to reach your local GPU, expose the backend with a
tunnel (e.g. `cloudflared tunnel --url http://localhost:8000`) and set
`NEXT_PUBLIC_API_URL` in Vercel to that tunnel URL.

GPU requires the NVIDIA Container Toolkit on the host. To run CPU-only,
set `INFERENCE_DEVICE=cpu` in `docker-compose.yml` and remove the `deploy.resources` block.

## Adding a new model wrapper

1. Create `models/<name>.py` subclassing `BaseInference`, implement `_load` and `predict`.
2. If the model has a custom `nn.Module`, do NOT copy the class — bind-mount the
   original repo into `/workspace/refs/<name>` and `sys.path.insert` it in `_load`.
3. Register a factory in `registry.py`.

## What's not 1:1 with the benchmark CSV yet

The benchmark shows finetuned (FT) variants per model. Wrappers currently point at
the "best" checkpoint we could identify per model dir. If you want to also load
the non-finetuned variant of a model, add a second entry in `registry.py` pointing
to that checkpoint — the wrapper class is reusable.
