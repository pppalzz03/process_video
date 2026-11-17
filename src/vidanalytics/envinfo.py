from __future__ import annotations

import platform
import subprocess
import sys


def _sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def collect() -> dict:
    info = {"python": sys.version.split()[0], "machine": platform.machine(), "platform": platform.platform()}
    for mod in ("ultralytics", "torch", "cv2", "tensorrt", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["cuda_available"] = None
    ff = _sh(["ffmpeg", "-version"])
    info["ffmpeg"] = ff.splitlines()[0] if ff else None
    info["nvidia_smi"] = _sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    return info


def gpu_mem_used_mb() -> float | None:
    """nvidia-smi memory.used; may be N/A on unified-memory GB10 -> None (reported as not captured)."""
    out = _sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return float(out.splitlines()[0])
    except Exception:
        return None
