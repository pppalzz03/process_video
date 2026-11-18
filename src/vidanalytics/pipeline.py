"""Core loop: frame source -> cross-chunk batches -> ONE detector -> per-segment event files."""
from __future__ import annotations

import time
from typing import Iterable

from .events import Detection, make_event, SegmentEventSink
from .frames import Frame, SegmentEnd
from .inference import Detector


def _rescale(dets: list[Detection], scale: tuple[float, float]) -> list[Detection]:
    sx, sy = scale
    if sx == 1.0 and sy == 1.0:
        return dets
    return [Detection(d.class_id, d.class_name, d.confidence,
                      (d.bbox_xyxy[0] * sx, d.bbox_xyxy[1] * sy, d.bbox_xyxy[2] * sx, d.bbox_xyxy[3] * sy))
            for d in dets]


def run_pipeline(source: Iterable[Frame | SegmentEnd], detector: Detector | None, batch_size: int,
                 sink: SegmentEventSink, sampling: dict) -> dict:
    """detector=None => decode-only (measures decode throughput; no events)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    t_decode = t_infer = t_write = 0.0
    frames = batches = 0
    pending: list[Frame] = []
    model_name = detector.model_name if detector else "none"

    def flush():
        nonlocal t_infer, t_write, frames, batches
        if not pending:
            return
        if detector is not None:
            t0 = time.perf_counter()
            dets = detector.predict([f.image for f in pending])
            t_infer += time.perf_counter() - t0
            if len(dets) != len(pending):
                raise RuntimeError("detector returned wrong number of results")
        else:
            dets = [[] for _ in pending]
        t0 = time.perf_counter()
        for f, d in zip(pending, dets):          # results mapped back via the frame's own tag
            d = _rescale(d, f.bbox_scale)
            sink.write_frame(f.tag, [make_event(f.tag, x, i, model_name, sampling) for i, x in enumerate(d)])
        sink.flush()
        t_write += time.perf_counter() - t0
        frames += len(pending)
        batches += 1
        pending.clear()

    wall0 = time.perf_counter()
    it = iter(source)
    while True:
        t0 = time.perf_counter()
        try:
            item = next(it)
        except StopIteration:
            break
        t_decode += time.perf_counter() - t0
        if isinstance(item, SegmentEnd):
            # earlier frames of this segment may still sit in `pending`; sink finalizes once written
            sink.input_done(item.segment_id, item.error)
            continue
        sink.register_frame(item.tag.segment_id)
        pending.append(item)
        if len(pending) >= batch_size:
            flush()
    flush()
    wall = time.perf_counter() - wall0
    extra = source.stats() if hasattr(source, "stats") else {}
    return {**extra,
        "frames_processed": frames, "batches": batches,
        # decode_seconds = time the main loop WAITED for frames (in threaded mode: GPU-starved time)
        "avg_batch_size": round(frames / batches, 3) if batches else 0,
        "wall_seconds": round(wall, 4), "decode_seconds": round(t_decode, 4),
        "infer_seconds": round(t_infer, 4), "write_seconds": round(t_write, 4),
        # processed (post-stride) frames per wall second; NOT source/input FPS, NOT output-video FPS
        "processed_fps": round(frames / wall, 3) if wall > 0 else None,
        "infer_only_fps": round(frames / t_infer, 3) if t_infer > 0 else None,
        "decode_only_fps": round(frames / t_decode, 3) if t_decode > 0 else None,
    }
