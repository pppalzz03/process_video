"""Output-agreement metrics, used to check TensorRT FP16 against PyTorch FP32."""
from __future__ import annotations

from .events import Detection


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare_detections(ref: list[list[Detection]], test: list[list[Detection]], iou_thr: float = 0.5) -> dict:
    """Greedy same-class matching per frame (highest IoU first). `ref` is ground truth for agreement."""
    if len(ref) != len(test):
        raise ValueError("frame count mismatch")
    matched = ref_only = test_only = 0
    ious, cdiffs = [], []
    for rf, tf in zip(ref, test):
        pairs = sorted(((iou(r.bbox_xyxy, t.bbox_xyxy), i, j)
                        for i, r in enumerate(rf) for j, t in enumerate(tf) if r.class_id == t.class_id),
                       reverse=True)
        ur, ut = set(), set()
        for v, i, j in pairs:
            if v < iou_thr:
                break
            if i in ur or j in ut:
                continue
            ur.add(i); ut.add(j)
            ious.append(v); cdiffs.append(abs(rf[i].confidence - tf[j].confidence))
        matched += len(ur)
        ref_only += len(rf) - len(ur)
        test_only += len(tf) - len(ut)
    prec = matched / (matched + test_only) if matched + test_only else 1.0
    rec = matched / (matched + ref_only) if matched + ref_only else 1.0
    return {
        "iou_threshold": iou_thr, "matched": matched, "ref_only": ref_only, "test_only": test_only,
        "precision_vs_ref": round(prec, 4), "recall_vs_ref": round(rec, 4),
        "mean_matched_iou": round(sum(ious) / len(ious), 4) if ious else None,
        "mean_abs_conf_diff": round(sum(cdiffs) / len(cdiffs), 4) if cdiffs else None,
    }
