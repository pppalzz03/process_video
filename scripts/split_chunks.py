#!/usr/bin/env python3
"""Stream-copy split + manifest.csv. Segment lengths follow keyframes (not exactly N s)."""
import argparse, sys
from vidanalytics.chunking import split_video

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--segment-seconds", type=float, default=60.0)
ap.add_argument("--video-id", default=None)
a = ap.parse_args()
try:
    chunks = split_video(a.video, a.out_dir, a.segment_seconds, a.video_id)
except Exception as e:
    sys.exit(f"ERROR: {e}")
for c in chunks:
    print(f"{c.segment_id} start={c.start_seconds:.3f} end={c.end_seconds:.3f} frames={c.frame_count} {c.path}")
print(f"manifest: {a.out_dir}/manifest.csv")
