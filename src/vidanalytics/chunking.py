"""Optional stream-copy segmentation. Chunk lengths are keyframe-bound, NOT exactly N seconds."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from .manifest import Chunk, write_manifest
from .timestamps import probe_packet_pts


def split_video(video: str | Path, out_dir: str | Path, segment_seconds: float = 60.0,
                video_id: str | None = None) -> list[Chunk]:
    video, out_dir = Path(video), Path(out_dir)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = video_id or video.stem
    raw_list = out_dir / "_segments_raw.csv"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:v:0", "-an", "-c", "copy",
           "-f", "segment", "-segment_time", str(segment_seconds), "-reset_timestamps", "1",
           "-segment_list", str(raw_list), "-segment_list_type", "csv",
           str(out_dir / "chunk_%04d.mp4")]
    subprocess.run(cmd, check=True)

    # NOTE: ffmpeg's segment_list start/end are NOT usable as timeline anchors: with B-frames they
    # are shifted by the reorder delay (observed +0.2 s). Anchor on the SOURCE video's real packet
    # PTS instead, indexed by cumulative frame count (stream copy never drops/duplicates frames).
    src_pts = probe_packet_pts(video)
    names = []
    with open(raw_list, newline="") as fh:
        for row in csv.reader(fh):
            names.append(row[0])
    counts = [len(probe_packet_pts(out_dir / n)) for n in names]
    if sum(counts) != len(src_pts):
        raise RuntimeError(f"chunks hold {sum(counts)} frames but source has {len(src_pts)}; "
                           "cannot map chunk timestamps to the original timeline")
    interval = (src_pts[-1] - src_pts[-2]) if len(src_pts) > 1 else 0.0
    chunks, idx = [], 0
    for i, (name, n) in enumerate(zip(names, counts)):
        start = src_pts[idx]
        end = src_pts[idx + n] if idx + n < len(src_pts) else src_pts[-1] + interval
        chunks.append(Chunk(video_id, f"seg_{i:04d}", name, start, end, n, idx, str(video)))
        idx += n
    write_manifest(chunks, out_dir / "manifest.csv")
    return chunks
