import json
import threading
import time

import numpy as np
import pytest
from conftest import needs_ffmpeg
from vidanalytics.chunking import split_video
from vidanalytics.decoder import decode_chunks, letterbox_size
from vidanalytics.eventcmp import compare_event_files
from vidanalytics.events import Detection, SegmentEventSink, merge_events
from vidanalytics.frames import Frame, FrameTag, SegmentEnd
from vidanalytics.manifest import Chunk, read_manifest
from vidanalytics.parallel_decoder import ParallelDecoder
from vidanalytics.pipeline import run_pipeline


def test_letterbox_size_matches_ultralytics_rule():
    assert letterbox_size(2858, 1350, 640) == (640, 302)      # r = 640/2858
    assert letterbox_size(640, 640, 640) == (640, 640)
    assert letterbox_size(1920, 1080, 640) == (640, 360)
    assert letterbox_size(100, 1, 640) == (640, 6)


@pytest.fixture(scope="module")
def chunks(synth_video, tmp_path_factory):
    d = tmp_path_factory.mktemp("ch")
    split_video(synth_video, d, segment_seconds=4)
    return read_manifest(d / "manifest.csv")          # resolves chunk paths against the manifest dir


def collect(src):
    frames, ends = [], []
    for it in src:
        (frames if isinstance(it, Frame) else ends).append(it)
    return frames, ends


@needs_ffmpeg
def test_threaded_equals_sequential(chunks, tmp_path):
    seq_f, seq_e = collect(decode_chunks(chunks))
    par_f, par_e = collect(ParallelDecoder(chunks, workers=4, queue_frames=8))
    key = lambda f: (f.tag.segment_id, f.tag.segment_frame_index)
    assert sorted(map(key, par_f)) == sorted(map(key, seq_f))
    assert len(par_f) == 250 and len(par_e) == len(chunks) and all(e.error is None for e in par_e)
    by_key = {key(f): f for f in seq_f}
    for f in par_f:                                       # same pixels, same timestamps, same tags
        g = by_key[key(f)]
        assert f.tag == g.tag and np.array_equal(f.image, g.image)
    for sid in {f.tag.segment_id for f in par_f}:         # per-segment order preserved
        idx = [f.tag.segment_frame_index for f in par_f if f.tag.segment_id == sid]
        assert idx == sorted(idx)


@needs_ffmpeg
def test_segment_end_comes_after_all_its_frames(chunks):
    seen_end, late = set(), 0
    for it in ParallelDecoder(chunks, workers=3, queue_frames=4):
        if isinstance(it, SegmentEnd):
            seen_end.add(it.segment_id)
        elif it.tag.segment_id in seen_end:
            late += 1
    assert late == 0


@needs_ffmpeg
def test_early_close_stops_threads_without_hanging(chunks):
    before = {t.name for t in threading.enumerate()}
    it = iter(ParallelDecoder(chunks, workers=4, queue_frames=2))   # tiny queue => producers blocked
    for _ in range(3):
        next(it)
    t0 = time.time()
    it.close()
    assert time.time() - t0 < 5
    time.sleep(0.2)
    assert not [t for t in threading.enumerate() if t.name.startswith("decode-") and t.name not in before]


@needs_ffmpeg
def test_backpressure_bounds_queue(chunks):
    dec = ParallelDecoder(chunks, workers=4, queue_frames=5)
    n = 0
    for it in dec:
        n += 1
        time.sleep(0.001)                                  # slow consumer
    s = dec.stats()
    assert s["decoder_queue_peak_depth"] <= 5 and s["decoder_producer_blocked_seconds"] > 0


@needs_ffmpeg
def test_missing_chunk_isolated_in_threads(chunks, tmp_path):
    bad = Chunk("v", "seg_bad", str(tmp_path / "nope.mp4"), 0.0, 1.0, 10, 9999)
    frames, ends = collect(ParallelDecoder([bad] + chunks, workers=2, queue_frames=8))
    assert len(frames) == 250
    err = [e for e in ends if e.error]
    assert len(err) == 1 and err[0].segment_id == "seg_bad" and "not found" in err[0].error


@needs_ffmpeg
def test_resize_in_decoder_and_bbox_scale(chunks):
    frames, _ = collect(ParallelDecoder(chunks[:1], workers=1, resize_imgsz=64))
    f = frames[0]
    assert f.image.shape[:2] == (48, 64)                   # 160x120 -> letterbox size for imgsz 64
    assert f.bbox_scale == pytest.approx((160 / 64, 120 / 48))


@needs_ffmpeg
def test_max_frames_limit(chunks):
    frames, _ = collect(ParallelDecoder(chunks, workers=1, max_frames=10))
    assert len(frames) == 10


def test_pipeline_rescales_boxes_to_original_pixels(tmp_path):
    class Det:
        model_name = "fake"
        def predict(self, images):
            return [[Detection(0, "p", 0.9, (10.0, 20.0, 30.0, 40.0))] for _ in images]

    def src():
        tag = FrameTag("v", "s", 0, 0, 0.0, "pts")
        yield Frame(tag, np.zeros((4, 4, 3), np.uint8), bbox_scale=(2.0, 3.0))
        yield SegmentEnd("s")
    sink = SegmentEventSink(tmp_path)
    run_pipeline(src(), Det(), 2, sink, {})
    sink.close(); merge_events(tmp_path, ["s"])
    e = json.loads((tmp_path / "events.jsonl").read_text())
    assert e["bbox_xyxy"] == [20.0, 60.0, 60.0, 120.0]


def test_compare_event_files(tmp_path):
    def ev(frame, box, conf=0.9):
        return json.dumps({"video_id": "v", "frame_index": frame, "class_id": 0, "class_name": "p",
                           "confidence": conf, "bbox_xyxy": box})
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    a.write_text(ev(0, [0, 0, 10, 10]) + "\n" + ev(1, [5, 5, 20, 20]) + "\n")
    b.write_text(ev(1, [5, 5, 20, 20]) + "\n" + ev(0, [0, 0, 10, 10]) + "\n")           # same, other order
    c.write_text(ev(0, [0, 0, 10, 10]) + "\n" + ev(2, [1, 1, 9, 9]) + "\n")
    r = compare_event_files(a, b)
    assert r["identical"] and r["precision_vs_ref"] == r["recall_vs_ref"] == 1.0
    r = compare_event_files(a, c)
    assert not r["identical"] and (r["matched"], r["ref_only"], r["test_only"]) == (1, 1, 1)


@needs_ffmpeg
def test_worker_threads_cli_matches_sequential_decode_only(chunks, tmp_path):
    from vidanalytics import worker
    man = str(__import__("pathlib").Path(chunks[0].path).parent / "manifest.csv")
    for mode, extra in (("sequential", []), ("threads", ["--decode-workers", "3", "--queue-frames", "6"])):
        out = tmp_path / mode
        assert worker.main(["--manifest", man, "--decode-only", "--decoder", mode, "--batch", "5",
                            "--out", str(out)] + extra) == 0
        s = json.loads((out / "run_summary.json").read_text())
        assert s["stats"]["frames_processed"] == 250 and len(s["segments_completed"]) == len(chunks)
        assert s["decoder"]["mode"] == mode
    assert "decoder_queue_peak_depth" in s["stats"]
