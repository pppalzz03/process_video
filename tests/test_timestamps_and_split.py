import pytest
from conftest import needs_ffmpeg
from vidanalytics.chunking import split_video
from vidanalytics.manifest import read_manifest
from vidanalytics.timestamps import absolute_timestamps, probe_packet_pts, estimated_timestamps


def test_absolute_timestamps_anchor_and_spacing():
    ts = absolute_timestamps(60.0, [0.5, 0.6, 0.75])
    assert ts == pytest.approx([60.0, 60.1, 60.25])
    assert absolute_timestamps(1.0, []) == []


def test_estimated_timestamps():
    assert estimated_timestamps(2.0, 10, 3) == pytest.approx([2.0, 2.1, 2.2])


@needs_ffmpeg
def test_split_timeline_continuity_and_no_frame_loss(synth_video, tmp_path):
    chunks = split_video(synth_video, tmp_path, segment_seconds=5)
    assert len(chunks) >= 3
    assert any(abs((c.end_seconds - c.start_seconds) - 5) > 1e-3 for c in chunks[:-1])  # keyframe-bound, not exactly 5 s
    src = probe_packet_pts(synth_video)
    assert sum(c.frame_count for c in chunks) == len(src)          # nothing dropped / duplicated
    m = read_manifest(tmp_path / "manifest.csv")
    ts = []
    for c in m:
        ts += absolute_timestamps(c.start_seconds, probe_packet_pts(c.path))
    assert len(ts) == len(src)
    assert all(b > a for a, b in zip(ts, ts[1:]))                  # strictly increasing
    drift = max(abs((t - ts[0]) - (s - src[0])) for t, s in zip(ts, src))
    assert drift < 0.002, f"drift {drift}"
    assert [c.first_frame_index for c in m] == [sum(x.frame_count for x in m[:i]) for i in range(len(m))]
