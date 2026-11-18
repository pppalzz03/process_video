# Run on the DGX Spark HOST (not inside the container).
RUN      = docker compose run --rm yolo
VIDEO   ?= /videos/output.mp4
BATCHES ?= 8 16
FRAMES  ?= 256

build:        ; docker compose build
env:          ; $(RUN) python /app/scripts/inspect_env.py
probe:        ; $(RUN) python /app/scripts/probe_video.py $(VIDEO)
test:         ; $(RUN) python -m pytest /app/tests -q
# Phase 1
export-trt:   ; $(RUN) python /app/scripts/export_trt.py --weights yolo26n.pt --batches $(BATCHES) --out-dir /workspace/engines
phase1:       ; $(RUN) python /app/scripts/phase1_benchmark.py --video $(VIDEO) --weights yolo26n.pt --engines-dir /workspace/engines --batches $(BATCHES) --num-frames $(FRAMES) --out-dir /workspace/results
# Phase 2  (PyTorch FP16, batch 8: phase 1 showed TensorRT FP16 was not faster on this GPU)
split:        ; $(RUN) python /app/scripts/split_chunks.py $(VIDEO) --out-dir /workspace/chunks --segment-seconds 60
validate:     ; $(RUN) python /app/scripts/validate_timestamps.py /workspace/chunks/manifest.csv --source $(VIDEO)
phase2-decode: ; $(RUN) python -m vidanalytics.worker --manifest /workspace/chunks/manifest.csv --decode-only --out /workspace/runs/phase2_decode_only
phase2:       ; $(RUN) python -m vidanalytics.worker --manifest /workspace/chunks/manifest.csv --weights yolo26n.pt --batch 8 --out /workspace/runs/phase2_pt_fp16_b8
