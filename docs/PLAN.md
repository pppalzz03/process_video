# Plan and exit criteria

Work one phase at a time; do not start the next until the previous phase's exit criteria are met with measured evidence.

## Phase 1 — TensorRT FP16 at batch 8 / 16  (DONE: TensorRT not faster than PyTorch FP16 on this GPU; see README)
Run `make env && make export-trt && make phase1`.
Exit: engines build; batch 8 and 16 both run; agreement vs PyTorch FP32 acceptable on your footage; pick the batch with the best median FPS that does not hurt agreement. Record the table in README.

## Phase 2 — multi-chunk batching worker  (DONE: timestamps validated; decode is the bottleneck, 199 vs 591 FPS)
Run `make split`, `python scripts/validate_timestamps.py chunks/manifest.csv --source /videos/output.mp4`, then `make phase2` and a `--decode-only` run.
Exit: timestamps validated; events identical (same detections) for batch 8 vs 16 vs original sequential run; `run_summary.json` shows decode vs infer seconds. If `infer_seconds` ≪ `decode_seconds`, decoding is the bottleneck → phase 3.

## Phase 3 — parallel decoders + bounded queue  (code ready; run `make split SEG=10 ...`, `make phase3`, `make compare`)
Implemented as threads first (see README); processes + shared memory only if threads are not enough. Original notes: replace `decoder.decode_chunks` with N decoder workers (one per chunk) feeding a bounded `queue.Queue`/`multiprocessing` queue consumed by the same single inference loop (`pipeline.run_pipeline` already accepts any iterable of `Frame | SegmentEnd`). Add: backpressure, cancellation, queue idle-time metrics, ordering guarantees per segment. Verify NVDEC actually works on this ARM64 stack before assuming it.
Exit: higher measured `processed_fps` than phase 2 with identical events.

## Phase 4 — DeepStream (only if still throughput-bound after phase 3)
Separate experiment/branch; compare against the phase 3 numbers on the same footage and sampling.

## Phase 5 — events → Qdrant for LLM search
Local JSONL stays authoritative. Add semantic event summaries (embeddings) + structured payload (video_id, timestamps, class, confidence) into Qdrant; deterministic `event_id` makes upserts idempotent. Answers must cite timestamps. Tracking/duration/enter-leave logic is a separate step.
