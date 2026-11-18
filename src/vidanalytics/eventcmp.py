"""Compare two events.jsonl files (e.g. phase 2 sequential vs phase 3 threaded)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .events import Detection
from .metrics import compare_detections


def load_events(path: str | Path) -> dict[tuple[str, int], list[Detection]]:
    out: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)
            out[(e["video_id"], e["frame_index"])].append(
                Detection(e["class_id"], e["class_name"], e["confidence"], tuple(e["bbox_xyxy"])))
    return out


def compare_event_files(ref_path, test_path, iou_thr: float = 0.5) -> dict:
    """Frames with zero detections never appear in events.jsonl, so frames are the union of both files."""
    ref, test = load_events(ref_path), load_events(test_path)
    keys = sorted(set(ref) | set(test))
    r = [ref.get(k, []) for k in keys]
    t = [test.get(k, []) for k in keys]
    res = compare_detections(r, t, iou_thr)
    res.update({"frames_with_detections_ref": len(ref), "frames_with_detections_test": len(test),
                "detections_ref": sum(map(len, r)), "detections_test": sum(map(len, t)),
                "identical": r == t})
    return res
