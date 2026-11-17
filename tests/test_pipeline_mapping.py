import json
import numpy as np
from vidanalytics.events import Detection, SegmentEventSink, merge_events
from vidanalytics.frames import Frame, FrameTag, SegmentEnd, batched
from vidanalytics.pipeline import run_pipeline
from vidanalytics.inference import YoloDetector


def make_source(segments):
    """segments: {seg_id: n_frames}; image pixel value == global frame index (for mapping checks)."""
    g = 0
    for sid, n in segments.items():
        for i in range(n):
            tag = FrameTag("vid", sid, g, i, g * 0.1, "pts")
            yield Frame(tag, np.full((4, 4, 3), g, dtype=np.uint8))
            g += 1
        yield SegmentEnd(sid)


class FakeDetector:
    model_name = "fake"
    def __init__(self): self.batch_sizes = []
    def predict(self, images):
        self.batch_sizes.append(len(images))
        # one detection per frame whose x1 encodes the frame's own pixel value
        return [[Detection(0, "person", 0.9, (float(im[0, 0, 0]), 0, 1, 1))] for im in images]


def read_events(out):
    return [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]


def test_batches_span_chunks_and_results_map_back(tmp_path):
    segs = {"seg_0000": 5, "seg_0001": 7, "seg_0002": 3}       # 15 frames, batch 4 -> 4,4,4,3
    det, sink = FakeDetector(), SegmentEventSink(tmp_path)
    stats = run_pipeline(make_source(segs), det, 4, sink, {"vid_stride": 1})
    sink.close(); merge_events(tmp_path, list(segs))
    assert det.batch_sizes == [4, 4, 4, 3]
    assert stats["frames_processed"] == 15 and stats["batches"] == 4
    ev = read_events(tmp_path)
    assert len(ev) == 15
    for e in ev:                                                # detection belongs to ITS frame
        assert e["bbox_xyxy"][0] == e["frame_index"]
        assert e["timestamp_seconds"] == round(e["frame_index"] * 0.1, 6)
    assert [e["segment_id"] for e in ev] == ["seg_0000"] * 5 + ["seg_0001"] * 7 + ["seg_0002"] * 3
    assert sorted(sink.completed) == sorted(segs)
    assert not list((tmp_path / "events").glob("*.tmp"))


def test_event_ids_deterministic(tmp_path):
    ids = []
    for k in range(2):
        out = tmp_path / str(k)
        sink = SegmentEventSink(out)
        run_pipeline(make_source({"s": 3}), FakeDetector(), 2, sink, {})
        sink.close(); merge_events(out, ["s"])
        ids.append([e["event_id"] for e in read_events(out)])
    assert ids[0] == ids[1] and len(set(ids[0])) == 3


def test_failed_segment_is_isolated(tmp_path):
    def src():
        yield SegmentEnd("seg_0000", error="file not found")
        yield from make_source({"seg_0001": 2})
    sink = SegmentEventSink(tmp_path)
    run_pipeline(src(), FakeDetector(), 4, sink, {})
    assert "seg_0000" in sink.failed and sink.completed == ["seg_0001"]
    assert not (tmp_path / "events" / "seg_0000.jsonl").exists()


def test_resume_marker_is_final_file(tmp_path):
    sink = SegmentEventSink(tmp_path)
    run_pipeline(make_source({"a": 2}), FakeDetector(), 2, sink, {})
    assert SegmentEventSink(tmp_path).is_done("a") and not SegmentEventSink(tmp_path).is_done("b")


def test_decode_only_mode(tmp_path):
    sink = SegmentEventSink(tmp_path)
    stats = run_pipeline(make_source({"a": 5}), None, 2, sink, {})
    assert stats["frames_processed"] == 5 and stats["infer_seconds"] == 0


def test_batched_helper():
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_static_engine_padding_discards_padded_results():
    class R:  # minimal ultralytics-like result
        def __init__(self): self.speed = {"inference": 1.0}; self.boxes = None; self.names = {}
    class M:
        def predict(self, feed, **kw): self.n = len(feed); return [R() for _ in feed]
    d = object.__new__(YoloDetector)
    d.model, d.kw, d.pad_to_batch, d.last_speed, d.model_name = M(), {}, 8, [], "m.engine"
    out = d.predict([object()] * 3)
    assert d.model.n == 8 and len(out) == 3 and len(d.last_speed) == 3
