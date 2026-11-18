import json
import pytest
from vidanalytics.sampling import build_tracklets, catch_probability, load_frames, recommend_stride, report


def ev(frame, box, cls=0, name="person", conf=0.9, fps=60):
    return {"video_id": "v", "frame_index": frame, "timestamp_seconds": frame / fps, "class_id": cls,
            "class_name": name, "confidence": conf, "bbox_xyxy": box}


def write(tmp_path, events):
    p = tmp_path / "e.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def test_catch_probability_is_phase_independent_closed_form():
    assert catch_probability(list(range(120)), 60) == 1.0            # visible 2 s: always caught at 1 FPS
    assert catch_probability(list(range(200, 210)), 60) == pytest.approx(10 / 60)
    assert catch_probability([5], 1) == 1.0


def test_tracking_links_moving_object_and_splits_classes(tmp_path):
    evs = [ev(f, [f, 0, f + 50, 50]) for f in range(30)]                  # slowly moving person
    evs += [ev(f, [200, 200, 260, 260], cls=2, name="car") for f in range(10, 20)]
    by_frame, fps = load_frames(write(tmp_path, evs))
    assert fps == pytest.approx(60)
    tl = build_tracklets(by_frame)
    assert sorted((t.class_name, len(t.frames)) for t in tl) == [("car", 10), ("person", 30)]


def test_gap_tolerance_and_split(tmp_path):
    evs = [ev(f, [0, 0, 50, 50]) for f in list(range(0, 5)) + list(range(7, 12))]    # 2-frame gap -> same object
    assert len(build_tracklets(load_frames(write(tmp_path, evs))[0], max_gap=2)) == 1
    assert len(build_tracklets(load_frames(write(tmp_path, evs))[0], max_gap=1)) == 2  # gap too large -> split


def test_two_overlapping_people_stay_two_tracklets(tmp_path):
    evs = []
    for f in range(20):
        evs.append(ev(f, [0, 0, 50, 50]))
        evs.append(ev(f, [300, 300, 350, 350]))
    assert len(build_tracklets(load_frames(write(tmp_path, evs))[0])) == 2


def test_report_and_recommendation(tmp_path):
    evs = [ev(f, [0, 0, 50, 50]) for f in range(0, 180)]                         # 3 s object
    evs += [ev(f, [300, 300, 350, 350], cls=2, name="car") for f in range(500, 510)]   # 10-frame flash
    by_frame, fps = load_frames(write(tmp_path, evs))
    r = report(build_tracklets(by_frame), fps, [1, 12, 60], min_frames=3)
    rows = {x["stride"]: x for x in r["rows"]}
    assert rows[1]["expected_miss_rate"] == 0.0
    assert rows[60]["miss_by_class"]["person"] == 0.0 and rows[60]["miss_by_class"]["car"] == pytest.approx(1 - 10 / 60, abs=1e-3)
    assert rows[12]["miss_by_class"]["car"] == pytest.approx(1 - 10 / 12, abs=1e-3)   # 10 of 12 phases catch it
    assert recommend_stride(r["rows"], 0.0)["stride"] == 1


def test_flicker_filtered(tmp_path):
    evs = [ev(0, [0, 0, 50, 50]), ev(100, [0, 0, 50, 50])] + [ev(f, [300, 300, 350, 350]) for f in range(50, 80)]
    r = report(build_tracklets(load_frames(write(tmp_path, evs))[0]), 60, [1], min_frames=3)
    assert r["tracklets_total"] == 3 and r["tracklets_kept"] == 1
