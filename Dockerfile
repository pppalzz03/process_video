# Verify the tag exists for your DGX OS / CUDA version before building:
#   docker pull ultralytics/ultralytics:latest-nvidia-arm64
FROM ultralytics/ultralytics:latest-nvidia-arm64

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-trt.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
# --no-deps: never let pip/uv touch the base image's NVIDIA torch (see requirements-trt.txt)
RUN pip install --no-cache-dir --no-deps -r requirements-trt.txt \
 && python -c "import onnx_graphsurgeon, onnx, onnxruntime; print('trt export deps ok')"
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
# --no-deps: keep the base image's CUDA torch untouched
RUN pip install --no-cache-dir --no-deps -e .

# Ultralytics must not pip-install behind our back (it tries to re-pin torch and fails)
ENV PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=False
CMD ["bash"]
