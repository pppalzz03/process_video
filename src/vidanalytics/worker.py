"""Phase 2 worker: read many chunks -> one batched GPU worker -> JSONL events.

    python -m vidanalytics.worker --manifest chunks/manifest.csv --weights yolo26n.pt --batch 16 --out runs/p2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import PIPELINE_VERSION, envinfo
from .decoder import decode_chunks
from .events import SegmentEventSink, merge_events
from .manifest import Chunk, chunks_from_files, read_manifest
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="manifest.csv from split_chunks.py (preferred)")
    src.add_argument("--inputs", nargs="+", help="contiguous chunk files in order (no manifest)")
    p.add_argument("--video-id", default=None, help="only with --inputs; default: first file stem")
    p.add_argument("--weights", default="yolo26n.pt", help=".pt or a TensorRT .engine")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--pad-to-batch", type=int, default=None,
                   help="required for static-batch TensorRT engines (set to the engine batch)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--vid-stride", type=int, default=1, help="1 = every frame (correctness baseline)")
    p.add_argument("--no-half", action="store_true", help="FP32 for .pt weights")
    p.add_argument("--decode-only", action="store_true", help="no inference: measure decode throughput")
    p.add_argument("--no-resume", action="store_true", help="re-process segments that already finished")
    p.add_argument("--max-frames", type=int, default=None, help="debug limit; segments cut short still count as done, so use a fresh --out")
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a = build_parser().parse_args(argv)
    try:
        if a.manifest:
            chunks: list[Chunk] = read_manifest(a.manifest)
        else:
            vid = a.video_id or Path(a.inputs[0]).stem
            chunks = chunks_from_files(a.inputs, vid)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not chunks:
        print("ERROR: no chunks to process", file=sys.stderr)
        return 2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sink = SegmentEventSink(out)
    todo = chunks if a.no_resume else [c for c in chunks if not sink.is_done(c.segment_id)]
    skipped = [c.segment_id for c in chunks if c not in todo]

    detector = None
    if not a.decode_only:
        try:
            from .inference import YoloDetector
            detector = YoloDetector(a.weights, imgsz=a.imgsz, conf=a.conf, half=not a.no_half,
                                    pad_to_batch=a.pad_to_batch)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    sampling = {"vid_stride": a.vid_stride, "imgsz": a.imgsz, "conf": a.conf, "batch": a.batch}
    t0 = time.time()
    stats = run_pipeline(decode_chunks(todo, a.vid_stride, a.max_frames), detector, a.batch, sink, sampling)
    sink.close()
    merge_events(out, [c.segment_id for c in chunks])

    summary = {
        "pipeline_version": PIPELINE_VERSION, "mode": "decode-only" if a.decode_only else "infer",
        "weights": a.weights, "config": sampling, "pad_to_batch": a.pad_to_batch,
        "segments_total": len(chunks), "segments_completed": sink.completed,
        "segments_skipped_resume": skipped, "segments_failed": sink.failed,
        "started_unix": t0, "stats": stats, "env": envinfo.collect(),
        "note": "processed_fps = post-stride frames / wall seconds (decode+infer+write, sequential in phase 2)",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"stats": stats, "failed": sink.failed}, indent=2))
    return 1 if sink.failed else 0


if __name__ == "__main__":
    sys.exit(main())
