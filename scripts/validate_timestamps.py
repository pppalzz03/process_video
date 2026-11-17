#!/usr/bin/env python3
"""Check a manifest for timeline continuity: monotonic, no overlaps, no gaps > 1.5 frame intervals."""
import argparse, sys
from vidanalytics.manifest import read_manifest
from vidanalytics.timestamps import probe_packet_pts, absolute_timestamps

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--source", help="optional original video: compare chunk timestamps to its real PTS")
a = ap.parse_args()
chunks = read_manifest(a.manifest)
all_ts, problems = [], []
for c in chunks:
    ts = absolute_timestamps(c.start_seconds, probe_packet_pts(c.path))
    if len(ts) != c.frame_count:
        problems.append(f"{c.segment_id}: manifest frame_count {c.frame_count} != probed {len(ts)}")
    all_ts.extend(ts)
d = [b - x for x, b in zip(all_ts, all_ts[1:])]
if d:
    med = sorted(d)[len(d) // 2]
    if min(d) <= 0:
        problems.append(f"non-increasing timestamps (min delta {min(d):.6f})")
    if max(d) > 1.5 * med:
        problems.append(f"gap {max(d):.4f}s > 1.5x median frame interval {med:.4f}s")
if a.source:
    src = probe_packet_pts(a.source)
    if len(src) != len(all_ts):
        problems.append(f"source has {len(src)} frames, chunks {len(all_ts)}")
    else:
        err = max(abs((x - all_ts[0]) - (y - src[0])) for x, y in zip(all_ts, src)) if src else 0
        print(f"max |chunk_ts - source_pts| (after aligning first frame): {err:.6f}s")
        if err > 0.002:
            problems.append(f"timestamp drift vs source: {err:.6f}s")
print(f"frames={len(all_ts)} chunks={len(chunks)}")
print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
sys.exit(1 if problems else 0)
