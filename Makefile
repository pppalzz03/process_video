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
# Phase 2/3 knobs:  make split SEG=10 CHUNK_DIR=/workspace/chunks10 ; make phase3 CHUNK_DIR=... WORKERS=4 QFRAMES=64
SEG       ?= 60
CHUNK_DIR ?= /workspace/chunks
WORKERS   ?= 4
QFRAMES   ?= 64
P2OUT     ?= /workspace/runs/phase2_pt_fp16_b8
W = python -m vidanalytics.worker --manifest $(CHUNK_DIR)/manifest.csv --weights yolo26n.pt --batch 8

# Phase 2  (PyTorch FP16, batch 8: phase 1 showed TensorRT FP16 was not faster on this GPU)
split:        ; $(RUN) python /app/scripts/split_chunks.py $(VIDEO) --out-dir $(CHUNK_DIR) --segment-seconds $(SEG)
validate:     ; $(RUN) python /app/scripts/validate_timestamps.py $(CHUNK_DIR)/manifest.csv --source $(VIDEO)
phase2-decode: ; $(RUN) python -m vidanalytics.worker --manifest $(CHUNK_DIR)/manifest.csv --decode-only --out /workspace/runs/phase2_decode_only
phase2:       ; $(RUN) $(W) --out $(P2OUT)
# Phase 3  (threaded decoders + bounded queue)
phase3-decode: ; $(RUN) python -m vidanalytics.worker --manifest $(CHUNK_DIR)/manifest.csv --decode-only --decoder threads --decode-workers $(WORKERS) --queue-frames $(QFRAMES) --out /workspace/runs/phase3_decode_only_w$(WORKERS)
phase3:       ; $(RUN) $(W) --decoder threads --decode-workers $(WORKERS) --queue-frames $(QFRAMES) --out /workspace/runs/phase3_w$(WORKERS)
phase3-resize: ; $(RUN) $(W) --decoder threads --decode-workers $(WORKERS) --queue-frames $(QFRAMES) --decode-resize --out /workspace/runs/phase3_resize_w$(WORKERS)
compare:      ; $(RUN) python /app/scripts/compare_events.py $(P2OUT)/events.jsonl /workspace/runs/phase3_w$(WORKERS)/events.jsonl
compare-resize: ; $(RUN) python /app/scripts/compare_events.py $(P2OUT)/events.jsonl /workspace/runs/phase3_resize_w$(WORKERS)/events.jsonl

# Which approach is best? Stage A (no GPU, fast) then Stage B (full pipeline, only promising configs).
# Results: /workspace/sweep/sweep_<mode>_<time>.md   (keep CPU/GPU otherwise idle while this runs)
SWEEP_SEGS    ?= 10
SWEEP_WORKERS ?= 4 8
sweep-decode: ; $(RUN) python /app/scripts/sweep_phase3.py --video $(VIDEO) --out-dir /workspace/sweep --mode decode-only --segs 60 20 10 5 --workers 1 2 4 8 12 16 --resize both
sweep:        ; $(RUN) python /app/scripts/sweep_phase3.py --video $(VIDEO) --out-dir /workspace/sweep --mode infer --segs $(SWEEP_SEGS) --workers $(SWEEP_WORKERS) --resize both
