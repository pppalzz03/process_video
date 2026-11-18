"""Chunk decoding. `decode_chunk` is the unit of work; sequential and threaded drivers share it."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator

from .frames import Frame, FrameTag, SegmentEnd
from .manifest import Chunk
from .timestamps import absolute_timestamps, probe_packet_pts

log = logging.getLogger(__name__)


def letterbox_size(width: int, height: int, imgsz: int) -> tuple[int, int]:
    """(w, h) after Ultralytics' first letterbox step: scale by min(imgsz/h, imgsz/w), round."""
    r = min(imgsz / height, imgsz / width)
    return max(round(width * r), 1), max(round(height * r), 1)   # same as LetterBox.get_params


def decode_chunk(ch: Chunk, stride: int = 1, resize_imgsz: int | None = None,
                 should_stop: Callable[[], bool] = lambda: False) -> Iterator[Frame | SegmentEnd]:
    """Yield Frames (every `stride`-th decoded frame) then exactly one SegmentEnd.

    Failures never raise: they become SegmentEnd(error=...) so other chunks still run.
    resize_imgsz: pre-resize to the size Ultralytics' letterbox would produce (cv2.INTER_LINEAR, same
    rounding) so the model sees the same pixels, but decoder threads do the work and frames shrink
    ~20x. Boxes are mapped back to original pixels through Frame.bbox_scale.
    """
    import cv2  # lazy: keeps pure-logic tests importable without OpenCV

    p = Path(ch.path)
    if not p.is_file():
        yield SegmentEnd(ch.segment_id, error=f"file not found: {p}")
        return
    try:
        pts = probe_packet_pts(p)
    except Exception as e:  # ffprobe failure
        yield SegmentEnd(ch.segment_id, error=f"ffprobe failed: {e}")
        return
    ts = absolute_timestamps(ch.start_seconds, pts)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        yield SegmentEnd(ch.segment_id, error=f"cannot open: {p}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    err = None
    stopped = False
    try:
        while True:
            if should_stop():
                stopped = True
                break
            ok, img = cap.read()
            if not ok:
                break
            if i % stride == 0:
                if i < len(ts):
                    t, src = ts[i], "pts"
                else:  # decoder produced more frames than packets: flag, don't hide
                    t, src = ch.start_seconds + i / fps, "estimated"
                scale = (1.0, 1.0)
                if resize_imgsz:
                    h, w = img.shape[:2]
                    nw, nh = letterbox_size(w, h, resize_imgsz)
                    if (nw, nh) != (w, h):
                        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                        scale = (w / nw, h / nh)
                yield Frame(FrameTag(ch.video_id, ch.segment_id, ch.first_frame_index + i, i, t, src),
                            img, scale)
            i += 1
    except GeneratorExit:
        cap.release()
        raise
    except Exception as e:
        err = f"decode error at frame {i}: {e}"
    cap.release()
    if err is None and not stopped and i != len(pts):
        log.warning("%s: decoded %d frames but ffprobe reported %d packets", ch.segment_id, i, len(pts))
    yield SegmentEnd(ch.segment_id, error=err)


def decode_chunks(chunks: list[Chunk], stride: int = 1, max_frames: int | None = None,
                  resize_imgsz: int | None = None) -> Iterator[Frame | SegmentEnd]:
    """Phase-2 sequential driver: chunk after chunk, one thread."""
    emitted = 0

    def stop() -> bool:
        return max_frames is not None and emitted >= max_frames

    for ch in chunks:
        if stop():
            return
        for item in decode_chunk(ch, stride, resize_imgsz, stop):
            if isinstance(item, Frame):
                emitted += 1
            yield item
