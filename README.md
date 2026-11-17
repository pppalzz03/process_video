# dgx-yolo26-video-analytics

FFmpeg + YOLO26n video analytics on a single **NVIDIA DGX Spark (GB10, ARM64)**.
Design rule: **parallel decode → bounded queue → ONE batched GPU worker** (never many competing YOLO processes).
Every claim in this repo must come from a measured run; unexecuted numbers are marked **not measured**.

| Phase | Goal | Code status | Measured on DGX Spark |
|---|---|---|---|
| 1 | YOLO26n TensorRT FP16, batch 8 / 16 | ready | **not measured** |
| 2 | Python worker: many chunks → cross-chunk batches | ready, unit/integration-tested (CPU, synthetic video) | **not measured** (GPU) |
| 3 | Parallel FFmpeg decoders + bounded queue | planned | – |
| 4 | DeepStream (only if still throughput-bound) | planned | – |
| 5 | Detection events + timestamps → Qdrant for LLM search | planned (event schema + deterministic IDs already in place) | – |

See [docs/PLAN.md](docs/PLAN.md) for the exit criteria of each phase.

## Layout

```
src/vidanalytics/   timestamps · manifest · chunking · decoder · frames · inference · pipeline · events · metrics · worker
scripts/            inspect_env · probe_video · split_chunks · validate_timestamps · export_trt · phase1_benchmark
tests/              timestamp continuity · batch→result mapping · missing files · resume · metrics
results/            benchmark reports (json is git-ignored; commit the .md tables you want to keep)
```

## Paths

| | Host | Container |
|---|---|---|
| Input video (read-only) | `/home/dgx/Downloads/output.mp4` | `/videos/output.mp4` |
| Outputs (engines, chunks, runs, results) | `/home/dgx/Downloads/yolo-output` | `/workspace` |

Override with `VIDEO_DIR=... OUT_DIR=... docker compose run ...`. Run `docker` on the **host**.

## Quick start (on the DGX Spark host)

```bash
git clone <this repo> && cd dgx-yolo26-video-analytics
make build            # docker compose build  (verify the base image tag first, see Dockerfile)
make env              # GPU, torch.cuda, ultralytics, tensorrt, ffmpeg versions
make probe            # codec / fps / VFR hint / real PTS stats of /videos/output.mp4
make test             # sanity tests inside the container
```

If `make env` prints `cuda_available: false`, stop: benchmarks would be invalid.

## Phase 1 — TensorRT FP16, batch 8 and 16

```bash
make export-trt       # builds engines/yolo26n_fp16_b8_640.engine and ..._b16_640.engine  (static batch)
make phase1           # PyTorch FP32/FP16 vs TensorRT FP16, batch 8 and 16, 256 real frames, 5 repeats
```

Output: `results/phase1_<timestamp>.json` + `.md` table. What it measures:

- `e2e FPS` — frames / wall time of `predict()` (incl. Ultralytics pre/post-processing). **Decode excluded**: frames are pre-loaded, so this is GPU-worker capacity, not pipeline FPS.
- `infer ms/img` — Ultralytics' own `speed['inference']`.
- `agreement P/R` — TensorRT detections vs PyTorch FP32 (same class, IoU ≥ 0.5). Accept the engine only if this is close to 1.0 on your footage.
- Warm-up batches are excluded; median of the repeats is reported.

Notes: engines are tied to this GPU + TensorRT version (never copy them elsewhere). Static-batch engines accept exactly their batch size; the worker pads the last partial batch and discards padded results (`--pad-to-batch`). `export_trt.py --dynamic` builds one dynamic engine instead — then verify batch 8 *and* 16 really run. GB10 uses unified memory, so `nvidia-smi` memory may be N/A; the report says "not captured" rather than guessing. Confirm that `yolo26n` is supported by the installed Ultralytics (`make env`).

### Troubleshooting phase 1

- `No module named 'onnx_graphsurgeon'` / `Ultralytics requirement ['nvidia-modelopt[onnx]>=0.44'] not found ... torch==2.15.0a0+...nv26.10 unsatisfiable`:
  Ultralytics' auto-install tries to re-pin torch to the base image's local NVIDIA build, which no index serves, so nothing gets installed.
  The Dockerfile now installs the TensorRT-export dependencies itself (`requirements-trt.txt`, with torch pinned by a constraint so it cannot be replaced) and sets `YOLO_AUTOINSTALL=False`. Run `make build` once (look for `trt export deps ok` in the output), then `make export-trt`. If the error persists the container is running an old image: check with `docker compose run --rm yolo python -c "import onnx_graphsurgeon"`.
- `'half' is deprecated ... Use 'quantize'`: warning only. `export_trt.py` uses `quantize=16` (falls back to `half=True` on older Ultralytics). If you still see it, the container image has an old copy of the scripts (rebuild, or use the live-mount in `compose.yaml`).

## Phase 2 — multi-chunk batching worker

```bash
make split            # chunks/ + manifest.csv  (ffmpeg -c copy, video only)
make phase2           # worker with the b16 engine; or run python -m vidanalytics.worker --help
# without a GPU / to measure decoding alone:
docker compose run --rm yolo python -m vidanalytics.worker --manifest /workspace/chunks/manifest.csv --decode-only --out /workspace/runs/decode_only
```

How it works:

1. `manifest.csv` records, per chunk, its start on the **original timeline**, frame count and global first-frame index. Stream-copy chunks cut at keyframes, so lengths are *not* exactly 60 s and `chunk_index * 60` is never used.
2. Timestamps come from ffprobe packet PTS (not guessed from FPS). Frames get a `FrameTag{video_id, segment_id, frame_index, timestamp_seconds, ts_source}`; if a fallback estimate is ever used it is flagged `ts_source="estimated"`.
3. Frames from consecutive chunks are grouped into batches that **span chunk boundaries**, run through one model instance, and results are mapped back through each frame's own tag.
4. Events are written per segment to `events/<segment>.jsonl.tmp` and atomically renamed when the segment is complete → `Resume` = skip segments whose `.jsonl` exists; a crash never leaves duplicate events. `events.jsonl` is the merged output. A bad/missing chunk fails only that segment (exit code 1, listed in `run_summary.json`).
5. `run_summary.json` separates `decode_seconds`, `infer_seconds`, `write_seconds` (phase 2 is sequential, so these show whether decoding starves the GPU — the input to the phase 3 decision). `processed_fps` = post-stride frames ÷ wall seconds; it is **not** source FPS or output-video FPS.

Event fields: `event_id` (deterministic UUIDv5), `video_id`, `segment_id`, `timestamp_seconds`, `frame_index`, `segment_frame_index`, `ts_source`, `class_id`, `class_name`, `confidence`, `bbox_xyxy`, `model_name`, `pipeline_version`, `sampling`. These are frame-level detections, not tracked events.

`--vid-stride 1` (every frame) is the correctness baseline; changing the stride changes the amount of work and must not be reported as speed-up.

## Validation done in this repo's authoring environment

- `pytest`: 16 passed on CPU with a synthetic H.264 (B-frame) video — timeline continuity of split chunks vs source PTS (max drift 0.000 s), batch→result mapping across chunk boundaries, partial-batch padding, resume, missing manifest/input/weights/chunk handling.
- This caught a real issue: ffmpeg's `segment_list` start times are shifted by the B-frame reorder delay (+0.2 s observed), so chunk starts are derived from the source's real PTS instead.
- **No GPU was available there:** TensorRT export, Ultralytics inference and every FPS number are **not measured** yet.

## Results (fill in from `results/*.md`)

| Variant | Batch | e2e FPS | infer ms/img | Agreement P/R |
|---|---|---|---|---|
| PyTorch FP32 | 8 / 16 | not measured | not measured | reference |
| TensorRT FP16 | 8 | not measured | not measured | not measured |
| TensorRT FP16 | 16 | not measured | not measured | not measured |
