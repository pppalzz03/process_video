from vidanalytics.events import Detection
from vidanalytics.metrics import compare_detections, iou


def D(c, conf, box): return Detection(c, str(c), conf, box)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_agreement():
    ref = [[D(0, .9, (0, 0, 10, 10)), D(1, .8, (50, 50, 60, 60))], []]
    tst = [[D(0, .85, (0, 0, 10, 11)), D(2, .7, (50, 50, 60, 60))], [D(0, .5, (1, 1, 2, 2))]]
    r = compare_detections(ref, tst)
    assert (r["matched"], r["ref_only"], r["test_only"]) == (1, 1, 2)
    assert r["mean_abs_conf_diff"] == 0.05
