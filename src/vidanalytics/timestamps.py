"""Real (not guessed) frame timestamps via ffprobe packet PTS."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def probe_packet_pts(path: str | Path) -> list[float]:
    """Presentation timestamps (seconds) of every video packet, sorted ascending.

    Packets are listed in decode order; decoders emit frames in presentation order,
    so sorted PTS[i] is the timestamp of the i-th decoded frame (one packet = one frame
    for normal H.264/H.265 streams).
    """
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)])
    pts = []
    for line in out.splitlines():
        tok = line.strip().strip(",")
        if not tok or tok == "N/A":
            continue
        pts.append(float(tok))
    pts.sort()
    return pts


def probe_stream_info(path: str | Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    data = json.loads(out)
    v = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError(f"no video stream in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    return {
        "codec": v.get("codec_name"),
        "width": v.get("width"),
        "height": v.get("height"),
        "avg_frame_rate": v.get("avg_frame_rate"),
        "r_frame_rate": v.get("r_frame_rate"),
        # avg != r (nominal) is a hint of variable frame rate
        "maybe_vfr": v.get("avg_frame_rate") != v.get("r_frame_rate"),
        "duration": float(data["format"].get("duration", 0) or 0),
        "nb_frames": v.get("nb_frames"),
        "time_base": v.get("time_base"),
        "has_audio": has_audio,
    }


def absolute_timestamps(chunk_start: float, pts: list[float]) -> list[float]:
    """Map chunk-local PTS to the original video timeline.

    chunk_start is the chunk's start on the original timeline (from the split manifest);
    the chunk's first frame is anchored there, later frames keep their real PTS spacing.
    """
    if not pts:
        return []
    p0 = pts[0]
    return [chunk_start + (p - p0) for p in pts]


def estimated_timestamps(chunk_start: float, fps: float, n: int) -> list[float]:
    """Fallback only. Callers must flag these as ts_source='estimated'."""
    return [chunk_start + i / fps for i in range(n)]
