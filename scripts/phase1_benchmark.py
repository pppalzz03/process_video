#!/usr/bin/env python3
"""Phase 1: YOLO26n TensorRT FP16 vs PyTorch at batch 8 / 16 on REAL frames from your video.

Measures inference throughput only (decode is excluded: frames are pre-loaded into RAM), so the
numbers answer "how fast can the GPU worker go" and are NOT end-to-end pipeline FPS.

Reported per variant (warm-up excluded, `--repeats` timed runs):
  e2e_fps_median   frames / wall time of model.predict() incl. Ultralytics pre/post-processing
  infer_ms_per_img median of Ultralytics' own `speed['inference']` (pure engine time)
  agreement        detections vs the PyTorch FP32 reference (same-class, IoU>=0.5)
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import cv2
from vidanalytics import envinfo
from vidanalytics.inference import YoloDetector
from vidanalytics.metrics import compare_detections

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--weights", default="yolo26n.pt")
ap.add_argument("--engines-dir", default="engines")
ap.add_argument("--batches", type=int, nargs="+", default=[8, 16])
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--num-frames", type=int, default=256, help="rounded down to a multiple of every batch size")
ap.add_argument("--start-frame", type=int, default=0)
ap.add_argument("--warmup", type=int, default=3, help="warm-up batches (not timed)")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--skip-pytorch", action="store_true")
ap.add_argument("--out-dir", default="results")
a = ap.parse_args()

lcm = 1
for b in a.batches:
    lcm = lcm * b // __import__("math").gcd(lcm, b)
n = (a.num_frames // lcm) * lcm
if n == 0:
    sys.exit(f"--num-frames must be >= {lcm}")

cap = cv2.VideoCapture(a.video)
if not cap.isOpened():
    sys.exit(f"ERROR: cannot open video: {a.video}")
cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
frames = []
while len(frames) < n:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
if len(frames) < n:
    n = (len(frames) // lcm) * lcm
    frames = frames[:n]
    print(f"WARNING: video shorter than requested; using {n} frames")
if n == 0:
    sys.exit("ERROR: no frames decoded")
print(f"loaded {n} frames {frames[0].shape}")


def run(det, B):
    """one pass over all frames in batches of B; returns (wall_s, per-image inference ms list, dets)"""
    ms, dets = [], []
    t0 = time.perf_counter()
    for i in range(0, n, B):
        dets += det.predict(frames[i:i + B])
        ms += [s["inference"] for s in det.last_speed]
    return time.perf_counter() - t0, ms, dets


def bench(label, det, B):
    for i in range(a.warmup):
        det.predict(frames[(i * B) % n:(i * B) % n + B] or frames[:B])
    try:
        import torch; torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    walls, all_ms, dets = [], [], None
    for _ in range(a.repeats):
        w, ms, dets = run(det, B)
        walls.append(w); all_ms += ms
    fps = [n / w for w in walls]
    peak = None
    try:
        import torch; peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    r = {"variant": label, "batch": B, "frames_per_run": n, "repeats": a.repeats,
         "e2e_fps_median": round(statistics.median(fps), 2), "e2e_fps_min": round(min(fps), 2),
         "e2e_fps_max": round(max(fps), 2),
         "infer_ms_per_img_median": round(statistics.median(all_ms), 3),
         "torch_peak_alloc_mb": peak,   # TensorRT's own allocations are not visible here
         "gpu_mem_used_mb_after": envinfo.gpu_mem_used_mb()}
    print(json.dumps(r))
    return r, dets


results, ref = [], None
for B in a.batches:
    if not a.skip_pytorch:
        det = YoloDetector(a.weights, imgsz=a.imgsz, half=False)
        r, d = bench("pytorch_fp32", det, B); results.append(r)
        ref = ref or d                     # reference detections for agreement
        det = YoloDetector(a.weights, imgsz=a.imgsz, half=True)
        r, _ = bench("pytorch_fp16", det, B); results.append(r)
        del det
    eng = Path(a.engines_dir) / f"{Path(a.weights).stem}_fp16_b{B}_{a.imgsz}.engine"
    if not eng.exists():
        print(f"MISSING engine {eng} -> run scripts/export_trt.py first", file=sys.stderr)
        results.append({"variant": "tensorrt_fp16", "batch": B, "error": f"engine not found: {eng}"})
        continue
    try:
        det = YoloDetector(str(eng), imgsz=a.imgsz, half=True, pad_to_batch=B)
        r, d = bench("tensorrt_fp16", det, B)
        r["engine"] = str(eng)
        if ref is not None:
            r["agreement_vs_pytorch_fp32"] = compare_detections(ref, d)
        results.append(r)
    except Exception as e:
        results.append({"variant": "tensorrt_fp16", "batch": B, "error": str(e)})

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
(out / f"phase1_{stamp}.json").write_text(json.dumps(
    {"video": a.video, "weights": a.weights, "imgsz": a.imgsz, "env": envinfo.collect(),
     "metric_definitions": {"e2e_fps": "inferred frames / wall s of predict(); decode excluded",
                            "infer_ms": "Ultralytics speed['inference'] per image"},
     "results": results}, indent=2))
rows = ["| variant | batch | e2e FPS (median) | infer ms/img | agreement P/R vs FP32 |", "|---|---|---|---|---|"]
for r in results:
    if "error" in r:
        rows.append(f"| {r['variant']} | {r['batch']} | ERROR: {r['error']} | | |"); continue
    ag = r.get("agreement_vs_pytorch_fp32")
    rows.append(f"| {r['variant']} | {r['batch']} | {r['e2e_fps_median']} | {r['infer_ms_per_img_median']} | "
                f"{(str(ag['precision_vs_ref'])+' / '+str(ag['recall_vs_ref'])) if ag else '-'} |")
(out / f"phase1_{stamp}.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
