#!/usr/bin/env python3
"""Estimate how many objects a sparser sampling rate would miss, using a FULL-RATE events.jsonl.

    analyze_sampling.py runs/phase2_pt_fp16_b8/events.jsonl --min-conf 0.4 --target-miss 0.05
Uses the detections you already have (no inference). Objects the detector missed at full rate are not counted.
"""
import argparse, json, sys
from vidanalytics.sampling import build_tracklets, load_frames, recommend_stride, report

ap = argparse.ArgumentParser()
ap.add_argument("events")
ap.add_argument("--fps", type=float, default=None, help="source FPS (default: estimated from timestamps)")
ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60, 120])
ap.add_argument("--min-conf", type=float, default=0.4, help="ignore weaker detections (reduces flicker)")
ap.add_argument("--min-frames", type=int, default=3, help="ignore tracklets shorter than this many frames")
ap.add_argument("--iou", type=float, default=0.3)
ap.add_argument("--max-gap", type=int, default=2, help="frames an object may vanish and stay the same object")
ap.add_argument("--target-miss", type=float, default=0.05, help="acceptable share of objects missed")
ap.add_argument("--json", default=None, help="also write the report here")
a = ap.parse_args()
try:
    by_frame, est = load_frames(a.events, a.min_conf)
except FileNotFoundError as e:
    sys.exit(f"ERROR: {e}")
fps = a.fps or est
if not by_frame or not fps:
    sys.exit("ERROR: no usable detections / cannot determine FPS (pass --fps)")
r = report(build_tracklets(by_frame, a.iou, a.max_gap), fps, a.strides, a.min_frames)
print(f"source fps {r['source_fps']}, tracklets {r['tracklets_total']} (kept {r['tracklets_kept']} with >= {a.min_frames} frames)")
print(f"time objects stay in view (s): p10={r['visible_seconds']['p10']}  p50={r['visible_seconds']['p50']}  p90={r['visible_seconds']['p90']}\n")
print("| stride | sample FPS | expected objects missed | by class |\n|---|---|---|---|")
for row in r["rows"]:
    bc = ", ".join(f"{c} {v:.1%}" for c, v in row["miss_by_class"].items())
    print(f"| {row['stride']} | {row['sample_fps']} | {row['expected_miss_rate']:.1%} | {bc} |")
rec = recommend_stride(r["rows"], a.target_miss)
print(f"\nSparsest sampling within {a.target_miss:.0%} missed objects: " +
      (f"stride {rec['stride']} = {rec['sample_fps']} FPS" if rec else "none of the listed strides (use stride 1 or lower the target)"))
print("Caveat: counts only objects detected at full rate; tracklet linking is approximate.")
if a.json:
    open(a.json, "w").write(json.dumps({**r, "recommended": rec}, indent=2))
