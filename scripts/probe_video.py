#!/usr/bin/env python3
"""ffprobe summary of a video: codec, fps, VFR hint, duration, audio, and real PTS stats."""
import argparse, json, sys
from vidanalytics.timestamps import probe_stream_info, probe_packet_pts

ap = argparse.ArgumentParser()
ap.add_argument("video")
a = ap.parse_args()
try:
    info = probe_stream_info(a.video)
    pts = probe_packet_pts(a.video)
except Exception as e:
    sys.exit(f"ERROR: {e}")
deltas = [b - c for c, b in zip(pts, pts[1:])]
info["packets"] = len(pts)
if deltas:
    info["pts_delta_min"], info["pts_delta_max"] = round(min(deltas), 6), round(max(deltas), 6)
    info["pts_delta_irregular"] = (max(deltas) - min(deltas)) > 0.5 * (sum(deltas) / len(deltas))
print(json.dumps(info, indent=2))
