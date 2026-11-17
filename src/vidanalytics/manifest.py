"""Chunk manifest: the single source of truth for where each chunk sits on the original timeline."""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    video_id: str
    segment_id: str
    path: str
    start_seconds: float        # on the ORIGINAL video timeline (never chunk_index*60)
    end_seconds: float
    frame_count: int            # decoded-frame count from ffprobe packets
    first_frame_index: int      # global index of this chunk's first frame in the original video
    source_path: str = ""


FIELDS = [f.name for f in fields(Chunk)]


def write_manifest(chunks: list[Chunk], path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for c in chunks:
            w.writerow(asdict(c))


def read_manifest(path: str | Path) -> list[Chunk]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    base = p.parent
    out = []
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            chunk_path = Path(row["path"])
            if not chunk_path.is_absolute():
                chunk_path = base / chunk_path          # relative paths resolve against the manifest
            out.append(Chunk(
                video_id=row["video_id"], segment_id=row["segment_id"], path=str(chunk_path),
                start_seconds=float(row["start_seconds"]), end_seconds=float(row["end_seconds"]),
                frame_count=int(row["frame_count"]), first_frame_index=int(row["first_frame_index"]),
                source_path=row.get("source_path", "")))
    return out


def chunks_from_files(paths: list[str], video_id: str) -> list[Chunk]:
    """No manifest: treat files as consecutive chunks of one video, starting at t=0.

    Start times are accumulated from real packet PTS, so this is only valid if the files
    are contiguous and in order. Prefer a manifest produced by split_chunks.py.
    """
    from .timestamps import probe_packet_pts
    chunks, t, idx = [], 0.0, 0
    for i, p in enumerate(paths):
        if not Path(p).is_file():
            raise FileNotFoundError(f"input not found: {p}")
        pts = probe_packet_pts(p)
        dur = (pts[-1] - pts[0]) + ((pts[-1] - pts[-2]) if len(pts) > 1 else 0.0) if pts else 0.0
        chunks.append(Chunk(video_id, f"seg_{i:04d}", str(p), t, t + dur, len(pts), idx, str(p)))
        t += dur
        idx += len(pts)
    return chunks
