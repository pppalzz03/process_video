"""Frame-level detection events (JSONL). Tracked/high-level events are a later phase."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import PIPELINE_VERSION
from .frames import FrameTag

_NS = uuid.UUID("6f1c6f0e-8e53-4c25-9a0d-5d1f7a4b0b10")


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def make_event(tag: FrameTag, det: Detection, det_idx: int, model_name: str, sampling: dict) -> dict:
    return {
        # deterministic id => re-runs are idempotent and Qdrant upserts don't duplicate
        "event_id": str(uuid.uuid5(_NS, f"{tag.video_id}|{tag.frame_index}|{det_idx}|{model_name}")),
        "video_id": tag.video_id,
        "segment_id": tag.segment_id,
        "timestamp_seconds": round(tag.timestamp_seconds, 6),
        "frame_index": tag.frame_index,
        "segment_frame_index": tag.segment_frame_index,
        "ts_source": tag.ts_source,
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": round(float(det.confidence), 5),
        "bbox_xyxy": [round(float(v), 2) for v in det.bbox_xyxy],
        "model_name": model_name,
        "pipeline_version": PIPELINE_VERSION,
        "sampling": sampling,
    }


class SegmentEventSink:
    """One JSONL per segment, written to *.tmp and atomically renamed when the segment is complete.

    A segment is complete when its input ended AND every yielded frame has been written.
    Presence of <segment>.jsonl therefore == "done" => resume just skips those segments, and a
    crash mid-segment leaves only a .tmp that is rewritten from scratch (no duplicate events).
    """

    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir) / "events"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh: dict[str, object] = {}
        self._yielded: dict[str, int] = {}
        self._written: dict[str, int] = {}
        self._input_done: set[str] = set()
        self.completed: list[str] = []
        self.failed: dict[str, str] = {}

    def is_done(self, segment_id: str) -> bool:
        return (self.dir / f"{segment_id}.jsonl").exists()

    def register_frame(self, segment_id: str) -> None:
        if segment_id not in self._fh:
            self._fh[segment_id] = open(self.dir / f"{segment_id}.jsonl.tmp", "w")
            self._yielded[segment_id] = 0
            self._written[segment_id] = 0
        self._yielded[segment_id] += 1

    def write_frame(self, tag: FrameTag, events: list[dict]) -> None:
        fh = self._fh[tag.segment_id]
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._written[tag.segment_id] += 1
        self._maybe_finalize(tag.segment_id)

    def flush(self) -> None:
        for fh in self._fh.values():
            fh.flush()

    def input_done(self, segment_id: str, error: str | None = None) -> None:
        if error:
            self.failed[segment_id] = error
            fh = self._fh.pop(segment_id, None)
            if fh:
                fh.close()  # leave .tmp behind for inspection; never promoted to .jsonl
            return
        self._input_done.add(segment_id)
        if segment_id not in self._fh:   # zero sampled frames: still a valid, empty, done segment
            (self.dir / f"{segment_id}.jsonl").write_text("")
            self.completed.append(segment_id)
            return
        self._maybe_finalize(segment_id)

    def _maybe_finalize(self, sid: str) -> None:
        if sid in self._input_done and sid in self._fh and self._written[sid] == self._yielded[sid]:
            self._fh.pop(sid).close()
            os.replace(self.dir / f"{sid}.jsonl.tmp", self.dir / f"{sid}.jsonl")
            self.completed.append(sid)

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()


def merge_events(out_dir: str | Path, segment_order: list[str]) -> Path:
    out = Path(out_dir) / "events.jsonl"
    ev = Path(out_dir) / "events"
    with open(out, "w") as w:
        for sid in segment_order:
            f = ev / f"{sid}.jsonl"
            if f.exists():
                w.write(f.read_text())
    return out
