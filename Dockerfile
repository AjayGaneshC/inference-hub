FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        libgl1 libglib2.0-0 \
        fonts-dejavu-core \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3

WORKDIR /workspace

# torch / torchvision with CUDA 12.8 wheels — required for Blackwell (sm_120, RTX 5090).
# Earlier cu121 wheels error with "no kernel image available" on a 5090.
RUN pip install --no-cache-dir \
        torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
