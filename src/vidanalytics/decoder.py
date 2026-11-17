"""Phase-2 decoder: sequential OpenCV decode per chunk. (Phase 3 swaps this for parallel FFmpeg.)"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from .frames import Frame, FrameTag, SegmentEnd
from .manifest import Chunk
from .timestamps import absolute_timestamps, estimated_timestamps, probe_packet_pts

log = logging.getLogger(__name__)


def decode_chunks(chunks: list[Chunk], stride: int = 1, max_frames: int | None = None
                  ) -> Iterator[Frame | SegmentEnd]:
    """Yield Frames (every `stride`-th decoded frame) then a SegmentEnd per chunk.

    Failures never raise: they become SegmentEnd(error=...) so other chunks still run.
    """
    import cv2  # lazy: keeps pure-logic tests importable without OpenCV

    emitted = 0
    for ch in chunks:
        if max_frames is not None and emitted >= max_frames:
            return
        p = Path(ch.path)
        if not p.is_file():
            yield SegmentEnd(ch.segment_id, error=f"file not found: {p}")
            continue
        try:
            pts = probe_packet_pts(p)
        except Exception as e:  # ffprobe failure
            yield SegmentEnd(ch.segment_id, error=f"ffprobe failed: {e}")
            continue
        ts = absolute_timestamps(ch.start_seconds, pts)
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            yield SegmentEnd(ch.segment_id, error=f"cannot open: {p}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        i = 0
        err = None
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    if i < len(ts):
                        t, src = ts[i], "pts"
                    else:  # decoder produced more frames than packets: flag, don't hide
                        t, src = ch.start_seconds + i / fps, "estimated"
                    yield Frame(FrameTag(ch.video_id, ch.segment_id, ch.first_frame_index + i,
                                         i, t, src), img)
                    emitted += 1
                    if max_frames is not None and emitted >= max_frames:
                        break
                i += 1
        except Exception as e:
            err = f"decode error at frame {i}: {e}"
        finally:
            cap.release()
        if err is None and max_frames is None and i != len(pts):
            log.warning("%s: decoded %d frames but ffprobe reported %d packets", ch.segment_id, i, len(pts))
        yield SegmentEnd(ch.segment_id, error=err)
