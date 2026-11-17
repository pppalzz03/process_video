import shutil, subprocess
import pytest

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg/ffprobe not installed")


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory):
    """25 s, 10 fps, H.264 with B-frames and a keyframe every 2 s (testsrc, no external media)."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    pytest.importorskip("cv2")
    p = tmp_path_factory.mktemp("vid") / "synth.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                    "-t", "25", "-c:v", "libx264", "-g", "20", "-bf", "2", "-pix_fmt", "yuv420p", str(p)],
                   check=True)
    return p
