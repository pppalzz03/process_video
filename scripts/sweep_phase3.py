#!/usr/bin/env python3
"""Sweep chunk length x decode workers x resize-in-decoder to find the best phase-3 setup.

Stage A (no GPU, fast):   sweep_phase3.py --video V --mode decode-only
Stage B (full pipeline):  sweep_phase3.py --video V --mode infer --segs 10 --workers 4 8 --resize both
Each config runs `--warmup-runs` (discarded: warms the OS file cache) + `--repeats` timed runs in a fresh
process; medians are reported. In infer mode every run's events are compared to the first (sequential,
no-resize) run, so a fast config that changes detections is visible.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

from vidanalytics import envinfo
from vidanalytics.chunking import split_video
from vidanalytics.eventcmp import compare_event_files
from vidanalytics.manifest import read_manifest
from vidanalytics.sweep import expand_configs, recommend, render_markdown, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out-dir", default="sweep")
ap.add_argument("--mode", choices=["decode-only", "infer"], default="decode-only")
ap.add_argument("--segs", type=float, nargs="+", default=[60, 20, 10, 5], help="chunk lengths in seconds")
ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
ap.add_argument("--resize", choices=["off", "on", "both"], default="off")
ap.add_argument("--weights", default="yolo26n.pt")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--queue-frames", type=int, default=64)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--warmup-runs", type=int, default=1)
ap.add_argument("--keep-chunks", action="store_true", help="do not delete chunk dirs at the end")
a = ap.parse_args()

if not Path(a.video).is_file():
    sys.exit(f"ERROR: video not found: {a.video}")
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
resize_opts = {"off": [False], "on": [True], "both": [False, True]}[a.resize]

# 1) chunk the video once per requested length
n_chunks, manifests = {}, {}
for seg in a.segs:
    d = out / f"chunks_seg{seg:g}"
    if not (d / "manifest.csv").exists():
        print(f"splitting into ~{seg:g}s chunks -> {d}", flush=True)
        split_video(a.video, d, seg)
    manifests[seg] = d / "manifest.csv"
    n_chunks[seg] = len(read_manifest(manifests[seg]))
print("chunks per length:", {f"{k:g}s": v for k, v in n_chunks.items()}, flush=True)

# 2) run every config
configs = expand_configs(a.segs, a.workers, resize_opts, n_chunks)
rows, ref_events = [], None
for i, c in enumerate(configs, 1):
    tag = f"seg{c['seg']:g}_{c['decoder']}_w{c['workers']}_{'rs' if c['resize'] else 'nors'}"
    cmd = [sys.executable, "-m", "vidanalytics.worker", "--manifest", str(manifests[c["seg"]]),
           "--batch", str(a.batch), "--decoder", c["decoder"], "--no-resume"]
    if c["decoder"] == "threads":
        cmd += ["--decode-workers", str(c["workers"]), "--queue-frames", str(a.queue_frames)]
    if c["resize"]:
        cmd += ["--decode-resize"]
    cmd += ["--decode-only"] if a.mode == "decode-only" else ["--weights", a.weights]
    stats, err, events = [], None, None
    for k in range(a.warmup_runs + a.repeats):
        run_dir = out / "runs" / f"{tag}_{k}"
        p = subprocess.run(cmd + ["--out", str(run_dir)], capture_output=True, text=True)
        if p.returncode != 0:
            err = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["worker failed"]
            err = err[0][:120]
            break
        s = json.loads((run_dir / "run_summary.json").read_text())
        if k >= a.warmup_runs:
            stats.append(s["stats"])
        events = run_dir / "events.jsonl"
        if k < a.warmup_runs + a.repeats - 1:
            shutil.rmtree(run_dir, ignore_errors=True)
    row = {**c, "chunks": n_chunks[c["seg"]], **(summarize(stats) if stats else {})}
    if err:
        row["error"] = err
    elif a.mode == "infer":
        if ref_events is None:
            ref_events = out / "reference_events.jsonl"
            shutil.copy(events, ref_events)
        else:
            row["agreement"] = compare_event_files(ref_events, events)
    rows.append(row)
    print(f"[{i}/{len(configs)}] {tag}: " + (f"FAILED {err}" if err else
          f"{row['processed_fps']} FPS (cores {row['avg_cores_used']})"), flush=True)

# 3) report
rec = recommend(rows)
md = render_markdown(rows, a.mode)
if rec:
    b, k = rec["best"], rec["knee"]
    md += (f"\nBest: seg={b['seg']:g}s, {b['decoder']}, workers={b['workers']}, resize={'on' if b['resize'] else 'off'} "
           f"-> {b['processed_fps']} FPS\nKnee (cheapest within 95% of best): seg={k['seg']:g}s, workers={k['workers']}, "
           f"resize={'on' if k['resize'] else 'off'} -> {k['processed_fps']} FPS\n")
    for n in rec["notes"]:
        md += f"Note: {n}\n"
stamp = time.strftime("%Y%m%d_%H%M%S")
(out / f"sweep_{a.mode}_{stamp}.json").write_text(json.dumps(
    {"video": a.video, "mode": a.mode, "args": vars(a), "env": envinfo.collect(), "nproc": os.cpu_count(),
     "rows": rows, "recommendation": rec}, indent=2, default=str))
(out / f"sweep_{a.mode}_{stamp}.md").write_text(md)
print("\n" + md)
if not a.keep_chunks:
    for seg in a.segs:
        shutil.rmtree(out / f"chunks_seg{seg:g}", ignore_errors=True)
shutil.rmtree(out / "runs", ignore_errors=True)
sys.exit(1 if any("error" in r for r in rows) else 0)
