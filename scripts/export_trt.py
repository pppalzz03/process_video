#!/usr/bin/env python3
"""Phase 1a: export YOLO26n to TensorRT FP16, one STATIC-batch engine per requested batch size.

Engines are tied to this GPU + TensorRT version: never copy them to another machine.
Use --dynamic to build a single dynamic-batch engine instead (verify batch 8 AND 16 actually run).
"""
import argparse, json, shutil, sys, time
from pathlib import Path
from vidanalytics import envinfo

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="yolo26n.pt")
ap.add_argument("--batches", type=int, nargs="+", default=[8, 16])
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out-dir", default="engines")
ap.add_argument("--dynamic", action="store_true", help="one dynamic engine, max batch = max(--batches)")
ap.add_argument("--workspace", type=float, default=None, help="TensorRT workspace GiB (default: auto)")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
stem = Path(a.weights).stem
jobs = [(max(a.batches), True)] if a.dynamic else [(b, False) for b in a.batches]
for b, dyn in jobs:
    name = f"{stem}_fp16_{'dyn' if dyn else 'b'}{b}_{a.imgsz}.engine"
    dst = out / name
    if dst.exists() and not a.force:
        print(f"skip (exists): {dst}"); continue
    kw = dict(format="engine", half=True, imgsz=a.imgsz, batch=b, dynamic=dyn, device=0)
    if a.workspace:
        kw["workspace"] = a.workspace
    t0 = time.time()
    try:
        produced = YOLO(a.weights).export(**kw)
    except Exception as e:
        print(f"EXPORT FAILED batch={b}: {e}", file=sys.stderr); continue
    shutil.move(str(produced), dst)     # export always writes <weights>.engine; rename per batch
    dst.with_suffix(".engine.json").write_text(json.dumps(
        {"export_args": kw, "build_seconds": round(time.time() - t0, 1), "env": envinfo.collect()}, indent=2))
    print(f"built {dst} in {time.time()-t0:.0f}s")
