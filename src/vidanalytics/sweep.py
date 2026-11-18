"""Pure logic for the phase-3 sweep: config expansion, aggregation, recommendation, report."""
from __future__ import annotations

import statistics
from typing import Iterable

FIELDS = ["processed_fps", "wall_seconds", "decode_seconds", "infer_seconds", "infer_only_fps",
          "decoder_producer_blocked_seconds", "decoder_queue_peak_depth", "avg_cores_used"]


def expand_configs(segs: list[float], workers: list[int], resize_opts: list[bool],
                   n_chunks: dict[float, int]) -> list[dict]:
    """Sequential baseline on the first segment size only; threads for every (seg, workers, resize).

    Requested workers are capped at the chunk count (that is what ParallelDecoder does), and configs that
    collapse to the same effective setting are run once.
    """
    out, seen = [], set()
    for resize in resize_opts:
        out.append({"seg": segs[0], "decoder": "sequential", "workers_req": 0, "workers": 0, "resize": resize})
    for seg in segs:
        for w in workers:
            eff = min(w, n_chunks[seg])
            for resize in resize_opts:
                key = (seg, eff, resize)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"seg": seg, "decoder": "threads", "workers_req": w, "workers": eff,
                            "resize": resize})
    return out


def summarize(runs: list[dict]) -> dict:
    """Median of each stat over repeated runs (stats dicts from run_summary.json)."""
    row = {}
    for f in FIELDS:
        vals = [r[f] for r in runs if r.get(f) is not None]
        row[f] = round(statistics.median(vals), 3) if vals else None
    fps = [r["processed_fps"] for r in runs if r.get("processed_fps") is not None]
    row["fps_min"], row["fps_max"], row["n_runs"] = (min(fps), max(fps), len(fps)) if fps else (None, None, 0)
    return row


def recommend(rows: list[dict], tol: float = 0.95) -> dict:
    """best = highest median FPS; knee = cheapest threaded config within `tol` of best."""
    ok = [r for r in rows if r.get("processed_fps")]
    if not ok:
        return {}
    best = max(ok, key=lambda r: r["processed_fps"])
    near = [r for r in ok if r["decoder"] == "threads" and r["processed_fps"] >= tol * best["processed_fps"]]
    knee = min(near, key=lambda r: (r["workers"], -r["processed_fps"])) if near else best
    notes = []
    ceiling = best.get("infer_only_fps")
    if ceiling and best["processed_fps"] >= 0.9 * ceiling:
        notes.append("best config is within 10% of the GPU-only rate: GPU-bound, more decode workers will not help")
    wall, blocked, w = best.get("wall_seconds"), best.get("decoder_producer_blocked_seconds"), best.get("workers")
    if wall and blocked is not None and w and blocked / (w * wall) > 0.5:
        notes.append("decoder threads spent >50% of their time blocked on a full queue: GPU-bound")
    return {"best": best, "knee": knee, "notes": notes}


def render_markdown(rows: list[dict], mode: str) -> str:
    seq = {r["resize"]: r["processed_fps"] for r in rows if r["decoder"] == "sequential" and r.get("processed_fps")}
    head = ["| seg s | chunks | decoder | workers | resize | FPS median | min-max | x vs sequential | cores used | "
            "blocked s | peak queue | vs reference (P/R) |", "|" + "---|" * 12]
    base = seq.get(False)
    lines = []
    for r in rows:
        if not r.get("processed_fps"):
            lines.append(f"| {r['seg']} | {r.get('chunks','')} | {r['decoder']} | {r['workers']} | "
                         f"{'on' if r['resize'] else 'off'} | FAILED: {r.get('error','')} | | | | | | |")
            continue
        sp = f"{r['processed_fps'] / base:.2f}" if base else "-"
        ag = r.get("agreement")
        agtxt = "-" if not ag else ("identical" if ag["identical"] else f"{ag['precision_vs_ref']}/{ag['recall_vs_ref']}")
        lines.append(f"| {r['seg']} | {r['chunks']} | {r['decoder']} | {r['workers']} | {'on' if r['resize'] else 'off'} | "
                     f"{r['processed_fps']} | {r['fps_min']}-{r['fps_max']} | {sp} | {r['avg_cores_used']} | "
                     f"{r.get('decoder_producer_blocked_seconds')} | {r.get('decoder_queue_peak_depth')} | {agtxt} |")
    return "\n".join([f"Mode: {mode}  (FPS = processed frames / wall seconds)", ""] + head + lines) + "\n"
