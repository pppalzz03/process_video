"""Phase 3: N decoder threads -> ONE bounded queue -> the single inference loop.

Why threads: cv2.VideoCapture.read() and cv2.resize() release the GIL, and threads share memory, so
11 MB (2858x1350) frames are never pickled/copied across processes. If threads prove insufficient,
switch to processes + shared memory; the consumer side (an iterator of Frame | SegmentEnd) does not change.

Guarantees
- Frames of ONE segment keep their order (a segment is decoded by exactly one thread, FIFO queue).
- Different segments interleave freely; every Frame/SegmentEnd carries its own tag, so batching and
  result mapping do not depend on arrival order across segments.
- Backpressure: the queue is bounded (`queue_frames`); producers block when the GPU is the bottleneck.
- Cancellation: closing the iterator (break / exception in the consumer) stops and joins all threads.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Iterator

from .decoder import decode_chunk
from .frames import Frame, SegmentEnd
from .manifest import Chunk

_DONE = object()


class _WorkerError:
    def __init__(self, exc: BaseException):
        self.exc = exc


class ParallelDecoder:
    def __init__(self, chunks: list[Chunk], stride: int = 1, workers: int = 4, queue_frames: int = 64,
                 max_frames: int | None = None, resize_imgsz: int | None = None):
        if workers < 1 or queue_frames < 1:
            raise ValueError("workers and queue_frames must be >= 1")
        self.chunks, self.stride, self.max_frames, self.resize_imgsz = chunks, stride, max_frames, resize_imgsz
        self.workers = min(workers, max(1, len(chunks)))
        self.queue_frames = queue_frames
        self._lock = threading.Lock()
        self._emitted = 0
        self._blocked_s = 0.0          # summed over threads: time spent waiting on a FULL queue
        self._frames_out = 0
        self._peak_depth = 0
        self._wall_s = 0.0

    def stats(self) -> dict:
        return {"decoder": "threads", "decode_workers": self.workers, "queue_frames": self.queue_frames,
                # high blocked time => producers wait for the GPU loop (GPU-bound); ~0 => decode-bound
                "decoder_producer_blocked_seconds": round(self._blocked_s, 4),
                "decoder_queue_peak_depth": self._peak_depth}

    def __iter__(self) -> Iterator[Frame | SegmentEnd]:
        q: queue.Queue = queue.Queue(maxsize=self.queue_frames)
        tasks: queue.Queue = queue.Queue()
        for c in self.chunks:
            tasks.put(c)
        stop = threading.Event()

        def should_stop() -> bool:
            if stop.is_set():
                return True
            return self.max_frames is not None and self._emitted >= self.max_frames

        def put(item) -> bool:
            t0 = time.perf_counter()
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            else:
                return False
            waited = time.perf_counter() - t0
            with self._lock:
                self._blocked_s += waited
                self._peak_depth = max(self._peak_depth, q.qsize())
            return True

        def worker():
            try:
                while not stop.is_set():
                    try:
                        ch = tasks.get_nowait()
                    except queue.Empty:
                        return
                    if self.max_frames is not None and self._emitted >= self.max_frames:
                        return
                    gen = decode_chunk(ch, self.stride, self.resize_imgsz, should_stop)
                    try:
                        for item in gen:
                            if isinstance(item, Frame):
                                with self._lock:
                                    self._emitted += 1
                            if not put(item):
                                return
                    finally:
                        gen.close()
            except BaseException as e:      # unexpected bug: surface it in the consumer, don't hang
                put(_WorkerError(e))
            finally:
                put(_DONE)

        threads = [threading.Thread(target=worker, name=f"decode-{i}", daemon=True)
                   for i in range(self.workers)]
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        finished = 0
        try:
            while finished < len(threads):
                item = q.get()
                if item is _DONE:
                    finished += 1
                elif isinstance(item, _WorkerError):
                    raise RuntimeError("decoder thread crashed") from item.exc
                else:
                    yield item
        finally:
            stop.set()
            try:                          # unblock producers stuck in put()
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            for t in threads:
                t.join(timeout=10)
            self._wall_s = time.perf_counter() - t_start
