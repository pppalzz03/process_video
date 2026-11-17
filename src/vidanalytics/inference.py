from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .events import Detection


class Detector(Protocol):
    model_name: str
    def predict(self, images: list) -> list[list[Detection]]: ...


class YoloDetector:
    """One model instance on one GPU; takes a list of BGR frames as ONE batch.

    pad_to_batch: TensorRT engines exported with a static batch only accept exactly that
    batch size. Partial batches (end of data) are padded by repeating the last frame and the
    padded results are discarded, so results always map 1:1 to the input frames.
    """

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.7,
                 half: bool = True, device: int | str = 0, pad_to_batch: int | None = None,
                 max_det: int = 300):
        if not Path(weights).is_file() and not weights.startswith("yolo"):
            raise FileNotFoundError(f"weights not found: {weights}")
        from ultralytics import YOLO  # lazy
        self.model = YOLO(weights, task="detect")
        self.model_name = Path(weights).name
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, device=device, max_det=max_det, verbose=False)
        # Precision: TensorRT engines have it baked in, so pass nothing. For .pt weights, Ultralytics
        # >= 8.4.8x wants `quantize` (16 / 32); `half` is deprecated there and only warns.
        if Path(weights).suffix != ".engine":
            from ultralytics.utils import DEFAULT_CFG_DICT
            if "quantize" in DEFAULT_CFG_DICT:
                self.kw["quantize"] = 16 if half else 32
            else:
                self.kw["half"] = half
        self.pad_to_batch = pad_to_batch
        self.last_speed: list[dict] = []   # per-image ms: preprocess / inference / postprocess

    def predict(self, images: list) -> list[list[Detection]]:
        n = len(images)
        if n == 0:
            return []
        feed = list(images)
        if self.pad_to_batch and n < self.pad_to_batch:
            feed += [images[-1]] * (self.pad_to_batch - n)
        results = self.model.predict(feed, **self.kw)
        if len(results) != len(feed):
            raise RuntimeError(f"model returned {len(results)} results for {len(feed)} inputs")
        self.last_speed = [dict(r.speed) for r in results[:n]]
        out = []
        for r in results[:n]:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append([])
                continue
            xyxy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
            out.append([Detection(int(c), str(r.names[int(c)]), float(s), tuple(map(float, bb)))
                        for bb, s, c in zip(xyxy, cf, cl)])
        return out
