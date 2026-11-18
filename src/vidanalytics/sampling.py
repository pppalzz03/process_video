"""How sparsely can we sample frames before objects are missed?

Input: events.jsonl from a FULL-RATE run (--vid-stride 1). Detections are linked frame-to-frame into
tracklets (greedy IoU, same class). For a candidate stride s, an object is caught if at least one of
the frames it was detected in is sampled. Averaged over all sampling phases this is exactly
    P(catch) = (number of distinct residues of its frame indices mod s) / s
so the result does not depend on which frame the sampling starts at.

Limits: only objects the detector found at full rate are counted (it cannot see what the detector never
detects); tracklet linking is approximate (an object lost for > max_gap frames splits into two).
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import iou


@dataclass
class Tracklet:
    class_name: str
    last_bbox: tuple
    start_frame: int
    last_frame: int
    frames: list[int] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)


def load_frames(path: str | Path, min_conf: float = 0.0) -> tuple[dict[int, list[dict]], float | None]:
    """frame_index -> detections; also the source FPS estimated from (frame_index, timestamp) pairs."""
    by_frame: dict[int, list[dict]] = defaultdict(list)
    ts: dict[int, float] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)
            ts[e["frame_index"]] = e["timestamp_seconds"]
            if e["confidence"] >= min_conf:
                by_frame[e["frame_index"]].append(e)
    fps = None
    keys = sorted(ts)
    rates = [(ts[b] - ts[a]) / (b - a) for a, b in zip(keys, keys[1:]) if b > a and ts[b] > ts[a]]
    if rates:
        fps = 1.0 / statistics.median(rates)
    return by_frame, fps


def build_tracklets(by_frame: dict[int, list[dict]], iou_thr: float = 0.3, max_gap: int = 2) -> list[Tracklet]:
    """Greedy per-class linking. max_gap = frames an object may go undetected and still be the same one."""
    done: list[Tracklet] = []
    active: dict[int, list[Tracklet]] = defaultdict(list)       # class_id -> open tracklets
    for f in sorted(by_frame):
        for cid, tl in active.items():                           # retire stale tracklets
            keep = []
            for t in tl:
                (keep if f - t.last_frame <= max_gap + 1 else done).append(t)
            active[cid] = keep
        taken: set[int] = set()
        for d in sorted(by_frame[f], key=lambda x: -x["confidence"]):
            best, best_v = None, iou_thr
            for t in active[d["class_id"]]:
                if id(t) in taken:
                    continue
                v = iou(t.last_bbox, d["bbox_xyxy"])
                if v >= best_v:
                    best, best_v = t, v
            if best is None:
                best = Tracklet(d["class_name"], tuple(d["bbox_xyxy"]), f, f)
                active[d["class_id"]].append(best)
            taken.add(id(best))
            best.last_bbox, best.last_frame = tuple(d["bbox_xyxy"]), f
            best.frames.append(f)
            best.confs.append(d["confidence"])
    for tl in active.values():
        done.extend(tl)
    return done


def catch_probability(frames: list[int], stride: int) -> float:
    return len({f % stride for f in frames}) / stride


def report(tracklets: list[Tracklet], fps: float, strides: list[int], min_frames: int = 3) -> dict:
    kept = [t for t in tracklets if len(t.frames) >= min_frames]   # drop 1-2 frame flickers
    durs = sorted((t.frames[-1] - t.frames[0] + 1) / fps for t in kept)

    def pct(p: float) -> float | None:
        return round(durs[min(len(durs) - 1, int(p * len(durs)))], 3) if durs else None

    rows = []
    for s in strides:
        per_class: dict[str, list[float]] = defaultdict(list)
        for t in kept:
            per_class[t.class_name].append(catch_probability(t.frames, s))
        allp = [p for v in per_class.values() for p in v]
        rows.append({
            "stride": s, "sample_fps": round(fps / s, 3),
            "expected_miss_rate": round(1 - statistics.fmean(allp), 4) if allp else None,
            "miss_by_class": {c: round(1 - statistics.fmean(v), 4) for c, v in sorted(per_class.items())},
        })
    return {"source_fps": round(fps, 3), "tracklets_total": len(tracklets), "tracklets_kept": len(kept),
            "min_frames": min_frames,
            "visible_seconds": {"p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90)},
            "rows": rows}


def recommend_stride(rows: list[dict], target_miss: float = 0.05) -> dict | None:
    ok = [r for r in rows if r["expected_miss_rate"] is not None and r["expected_miss_rate"] <= target_miss]
    return max(ok, key=lambda r: r["stride"]) if ok else None
