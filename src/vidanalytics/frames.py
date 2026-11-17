from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Any


@dataclass(frozen=True)
class FrameTag:
    video_id: str
    segment_id: str
    frame_index: int            # global index in the original video (all decoded frames, pre-stride)
    segment_frame_index: int
    timestamp_seconds: float    # original-timeline PTS
    ts_source: str              # "pts" (ffprobe) | "estimated" (fallback, flagged)


@dataclass
class Frame:
    tag: FrameTag
    image: Any                  # BGR ndarray (H, W, 3)


@dataclass(frozen=True)
class SegmentEnd:
    """Marker emitted by a frame source after the last frame of a chunk."""
    segment_id: str
    error: str | None = None


def batched(items: Iterable, size: int) -> Iterator[list]:
    """Group an iterable into lists of `size`; last one may be partial. Spans chunk boundaries."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    buf = []
    for it in items:
        buf.append(it)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
