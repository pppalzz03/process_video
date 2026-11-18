#!/usr/bin/env python3
"""Check that two runs found the same detections (exit 1 if agreement is below the thresholds).

    compare_events.py runs/phase2/events.jsonl runs/phase3/events.jsonl
`identical: true` means bit-for-bit equal detections; small differences are expected between batch
compositions in FP16 and when --decode-resize changes the rounding of box coordinates.
"""
import argparse, json, sys
from vidanalytics.eventcmp import compare_event_files

ap = argparse.ArgumentParser()
ap.add_argument("reference")
ap.add_argument("test")
ap.add_argument("--iou", type=float, default=0.5)
ap.add_argument("--min-agreement", type=float, default=0.99, help="min precision AND recall vs reference")
a = ap.parse_args()
try:
    r = compare_event_files(a.reference, a.test, a.iou)
except FileNotFoundError as e:
    sys.exit(f"ERROR: {e}")
print(json.dumps(r, indent=2))
ok = r["precision_vs_ref"] >= a.min_agreement and r["recall_vs_ref"] >= a.min_agreement
print("OK" if ok else f"DIFFERENT (agreement below {a.min_agreement})")
sys.exit(0 if ok else 1)
