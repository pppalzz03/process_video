# Verify the tag exists for your DGX OS / CUDA version before building:
#   docker pull ultralytics/ultralytics:latest-nvidia-arm64
FROM ultralytics/ultralytics:latest-nvidia-arm64

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-trt.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
# TensorRT export deps (onnx_graphsurgeon, modelopt, ...). A constraint pins the installed NVIDIA torch
# so dependency resolution cannot replace it; the build fails loudly if torch changed anyway.
RUN TV=$(python -c "import importlib.metadata as m; print(m.version('torch'))") \
 && echo "torch==${TV}" > /tmp/torch-constraint.txt \
 && pip install --no-cache-dir -c /tmp/torch-constraint.txt -r requirements-trt.txt \
 && test "$(python -c "import importlib.metadata as m; print(m.version('torch'))")" = "${TV}" \
 && python -c "import numpy, onnx, onnxruntime, onnx_graphsurgeon; print('trt export deps ok; numpy', numpy.__version__)"
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
# --no-deps: keep the base image's CUDA torch untouched
RUN pip install --no-cache-dir --no-deps -e .

# Ultralytics must not pip-install behind our back (it tries to re-pin torch and fails)
ENV PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=False
CMD ["bash"]
