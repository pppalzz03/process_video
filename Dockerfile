# Verify the tag exists for your DGX OS / CUDA version before building:
#   docker pull ultralytics/ultralytics:latest-nvidia-arm64
FROM ultralytics/ultralytics:latest-nvidia-arm64

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
# --no-deps: keep the base image's CUDA torch untouched
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1
CMD ["bash"]
