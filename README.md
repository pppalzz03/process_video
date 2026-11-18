# dgx-yolo26-video-analytics

FFmpeg + YOLO26n video analytics on a single **NVIDIA DGX Spark (GB10, ARM64)**.
Design rule: **parallel decode → bounded queue → ONE batched GPU worker** (never many competing YOLO processes).
Every claim in this repo must come from a measured run; unexecuted numbers are marked **not measured**.

| Phase | Goal | Code status | Measured on DGX Spark |
|---|---|---|---|
| 1 | YOLO26n TensorRT FP16, batch 8 / 16 | done | measured (see Results) |
| 2 | Python worker: many chunks → cross-chunk batches | done | measured (see Results); timestamps validated |
| 3 | Parallel decoders (threads) + bounded queue | code ready, tested on CPU | **not measured** on DGX |
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

## Phase 3 — parallel decoders + bounded queue

Phase 2 showed decode is the bottleneck (see Results), so `--decoder threads` runs N decoder threads (one chunk each) feeding a bounded queue that the single inference loop consumes. Threads, not processes: `cv2` releases the GIL while reading/resizing and 2858x1350 BGR frames are ~11.6 MB each, which would be expensive to pickle across processes.

```bash
make split SEG=10 CHUNK_DIR=/workspace/chunks10        # more/shorter chunks => more parallelism (60 s chunks of a 180 s video = only 3)
make phase3 CHUNK_DIR=/workspace/chunks10 WORKERS=4 QFRAMES=64
make compare WORKERS=4                                   # phase 2 events vs phase 3 events (must say OK)
make phase3-resize CHUNK_DIR=/workspace/chunks10 WORKERS=4   # optional: resize inside the decoder threads
make compare-resize WORKERS=4
```

- Frames of one segment stay in order; different segments interleave; results are mapped back by each frame's own tag, so output does not depend on arrival order.
- `decoder_producer_blocked_seconds` high → producers wait for the GPU (GPU-bound); near 0 → decode-bound (add workers/chunks). `decoder_queue_peak_depth` shows how full the queue got. In threaded mode `decode_seconds` is the time the inference loop *waited* for frames (GPU starved).
- Queue memory ≈ `queue_frames × frame bytes` (11.6 MB at 2858x1350 → 64 frames ≈ 740 MB).
- `--decode-resize` replicates Ultralytics' first letterbox step (same size/rounding, `INTER_LINEAR`) in the decoder threads, so the model sees the same pixels, frames shrink ~20x, and boxes are mapped back to original pixels. Detections can differ slightly in box rounding: always run `make compare-resize`.
- `make compare` / `scripts/compare_events.py` exits 1 if precision or recall vs the reference drops below 0.99. `identical: true` = bit-for-bit equal.
- Sanity check only (2-core CPU sandbox, synthetic 2858x1350 video, decode-only): 153 FPS sequential → 200 FPS with 4 threads. **Not a DGX Spark number.**

### Choosing the setup: sweep (chunk length x decode workers x resize)

```bash
make sweep-decode                       # Stage A: decode only, no GPU, ~minutes. Finds the decode ceiling
make sweep SWEEP_SEGS=10 SWEEP_WORKERS="4 8"   # Stage B: full pipeline on the promising configs only
```

Each config runs one discarded warm-up run (warms the OS file cache) plus 3 timed runs, each in a fresh process; the report shows the median, min-max, speed-up vs the sequential baseline, average CPU cores used, producer-blocked seconds and peak queue depth. In Stage B every config's events are compared with the sequential run, so a faster setup that changes detections is flagged (`vs reference`). Output: `/workspace/sweep/sweep_<mode>_<time>.md|json`. It prints the **best** config and the **knee** (cheapest config within 95% of best), plus a note when the result is GPU-bound. Keep the machine otherwise idle while it runs. Chunk lengths are keyframe-bound; check keyframe spacing first (see Troubleshooting / `ffprobe -skip_frame nokey`).

### Choosing a sampling rate (how many frames to process)

Use the detections you already have from a full-rate run (`--vid-stride 1`, e.g. the phase 2 run) to estimate how many objects a sparser rate would miss; no new inference needed:

```bash
make analyze-sampling                 # uses $(P2OUT)/events.jsonl; options: --min-conf 0.4 --target-miss 0.05
```

Detections are linked into per-object tracklets (greedy IoU, same class); for each candidate stride the expected share of objects with no sampled frame is computed exactly (independent of the sampling phase), per class. The report also shows how long objects stay in view. It prints the sparsest stride within the target miss rate. Limits: objects the detector never found at full rate are not counted; tracklet linking is approximate; results are specific to this camera/footage. State the chosen stride wherever results are reported (it is stored in every event's `sampling` field).

## Validation done in this repo's authoring environment

- `pytest`: 38 passed on CPU with a synthetic H.264 (B-frame) video — timeline continuity of split chunks vs source PTS (max drift 0.000 s), batch→result mapping across chunk boundaries, partial-batch padding, resume, missing manifest/input/weights/chunk handling.
- This caught a real issue: ffmpeg's `segment_list` start times are shifted by the B-frame reorder delay (+0.2 s observed), so chunk starts are derived from the source's real PTS instead.
- **No GPU was available there:** TensorRT export, Ultralytics inference and every FPS number are **not measured** yet.

## Results (measured on the DGX Spark, one clip: 2858x1350, H.264, 60 fps, 180.75 s, 10,845 frames; single runs)

Phase 1 — inference only (decode excluded), 256 frames, median of 5 runs, `quantize`/FP16:

| Variant | Batch | e2e FPS | infer ms/img | Agreement P/R vs PyTorch FP32 |
|---|---|---|---|---|
| PyTorch FP32 | 8 | 520.56 | 0.891 | reference |
| PyTorch FP16 | 8 | 683.33 | 0.457 | not computed |
| TensorRT FP16 | 8 | 582.63 | 0.622 | 0.9272 / 0.9867 |
| PyTorch FP32 | 16 | 494.82 | 1.020 | reference |
| PyTorch FP16 | 16 | 690.35 | 0.476 | not computed |
| TensorRT FP16 | 16 | 554.67 | 0.666 | 0.9286 / 0.9858 |

Finding: TensorRT FP16 was **not faster** than PyTorch FP16 here, and batch 16 gave no gain over 8. Pre/post-processing (~1 ms/img) dominates GPU time (~0.46 ms/img).

Phase 2 — PyTorch FP16, batch 8, 60 s chunks, sequential decode → infer (all 10,845 frames, 0 failed segments):

| decode | infer | write | wall | processed FPS | infer-only FPS | decode FPS |
|---|---|---|---|---|---|---|
| 35.78 s | 18.34 s | 0.33 s | 54.48 s | 199.1 | 591.3 | 303.1 |

Finding: decode is ~66% of wall time and slower than the GPU → phase 3. (`processed FPS` = frames inferred ÷ wall seconds; source is 60 fps, so ≈3.3x real time.)

Phase 3 — **not measured** yet.
