import json
from conftest import needs_ffmpeg
from vidanalytics import worker
from vidanalytics.chunking import split_video
from vidanalytics.manifest import read_manifest, write_manifest


def test_missing_manifest_is_clear_error(tmp_path, capsys):
    rc = worker.main(["--manifest", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o"), "--decode-only"])
    assert rc == 2 and "manifest not found" in capsys.readouterr().err


def test_missing_inputs_is_clear_error(tmp_path, capsys):
    rc = worker.main(["--inputs", str(tmp_path / "x.mp4"), "--out", str(tmp_path / "o"), "--decode-only"])
    assert rc == 2 and "input not found" in capsys.readouterr().err


def test_missing_weights_is_clear_error(tmp_path, capsys):
    (tmp_path / "m.csv").write_text("")
    rc = worker.main(["--manifest", str(tmp_path / "m.csv"), "--out", str(tmp_path / "o"),
                      "--weights", str(tmp_path / "missing.engine")])
    assert rc == 2


@needs_ffmpeg
def test_end_to_end_decode_only_resume_and_missing_chunk(synth_video, tmp_path):
    chunks = split_video(synth_video, tmp_path / "ch", segment_seconds=6)
    man = str(tmp_path / "ch" / "manifest.csv")
    out = tmp_path / "run"
    assert worker.main(["--manifest", man, "--decode-only", "--batch", "7", "--out", str(out)]) == 0
    s = json.loads((out / "run_summary.json").read_text())
    assert s["stats"]["frames_processed"] == 250 == sum(c.frame_count for c in chunks)
    assert len(s["segments_completed"]) == len(chunks)
    # resume: everything already done
    assert worker.main(["--manifest", man, "--decode-only", "--out", str(out)]) == 0
    s2 = json.loads((out / "run_summary.json").read_text())
    assert s2["stats"]["frames_processed"] == 0 and len(s2["segments_skipped_resume"]) == len(chunks)
    # a vanished chunk fails only that segment and yields exit code 1
    ms = read_manifest(man)
    (tmp_path / "ch" / ms[1].path.split("/")[-1]).unlink()
    out2 = tmp_path / "run2"
    assert worker.main(["--manifest", man, "--decode-only", "--out", str(out2)]) == 1
    s3 = json.loads((out2 / "run_summary.json").read_text())
    assert list(s3["segments_failed"]) == [ms[1].segment_id]
    assert s3["stats"]["frames_processed"] == 250 - ms[1].frame_count
