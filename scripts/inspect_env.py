#!/usr/bin/env python3
"""Print host/container capabilities relevant to the pipeline (run inside the container)."""
import json, os, shutil
from vidanalytics import envinfo

info = envinfo.collect()
info["in_container"] = os.path.exists("/.dockerenv")
info["ffprobe"] = shutil.which("ffprobe")
info["videos_mount_exists"] = os.path.isdir("/videos")
print(json.dumps(info, indent=2))
if not info.get("cuda_available"):
    print("\nWARNING: torch.cuda.is_available() is not True -> GPU benchmarks will not be valid here.")
