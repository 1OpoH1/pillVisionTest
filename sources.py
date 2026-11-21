from __future__ import annotations

import sys
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from mss import mss


class FrameSource:
    single_frame = False
    is_live = False
    supports_background = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    @property
    def name(self) -> str:
        return type(self).__name__


class ImageSource(FrameSource):
    single_frame = True
    supports_background = True

    def __init__(self, path: str):
        self.path = path
        self._frame = cv2.imread(path, cv2.IMREAD_COLOR)
        self._read = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._read or self._frame is None:
            return False, None
        self._read = True
        return True, self._frame.copy()

    @property
    def name(self) -> str:
        return f"image:{self.path}"


class VideoSource(FrameSource):
    is_live = True
    supports_background = True

    def __init__(self, path: str):
        self.path = path
        self._capture = cv2.VideoCapture(path)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._capture.read()
        return bool(ok), frame

    def release(self) -> None:
        self._capture.release()

    @property
    def name(self) -> str:
        return f"video:{self.path}"


@dataclass
class CameraMode:
    backend: int
    mjpg: bool
    pixels: int
    capture_fps: float
    total_fps: float


class CameraSource(FrameSource):
    is_live = True
    supports_background = True

    def __init__(self, index: int, benchmark: Optional[Callable[[np.ndarray], None]] = None):
        self.index = index
        self._capture: Optional[cv2.VideoCapture] = None
        self._first_frame: Optional[np.ndarray] = None
        self._original_threads: Optional[int] = None
        self.mode_info: Optional[str] = None
        if benchmark is None:
            self._capture = self._open(cv2.CAP_ANY, False)
        else:
            try:
                self._tune(benchmark)
            except BaseException:
                self.release()
                raise

    def _tune_threads(self, frame: np.ndarray, benchmark: Callable[[np.ndarray], None]) -> None:
        original = cv2.getNumThreads()
        self._original_threads = original
        timings = {}
        for threads in sorted({1, min(4, original), original}):
            cv2.setNumThreads(threads)
            benchmark(frame)
            samples = []
            for _ in range(3):
                started = perf_counter()
                benchmark(frame)
                samples.append(perf_counter() - started)
            timings[threads] = median(samples)
        fastest = min(timings.values())
        cv2.setNumThreads(min(threads for threads, elapsed in timings.items() if elapsed <= 1.05 * fastest))

    def _open(self, backend: int, mjpg: bool) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.index, backend)
        if capture.isOpened():
            if mjpg:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _tune(self, benchmark: Callable[[np.ndarray], None]) -> None:
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if sys.platform == "win32" else (cv2.CAP_ANY,)
        modes = []
        for backend in backends:
            for mjpg in (False, True):
                capture = self._open(backend, mjpg)
                try:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        continue
                    nominal_fps = capture.get(cv2.CAP_PROP_FPS)
                    started = perf_counter()
                    for _ in range(4):
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            break
                    capture_seconds = max((perf_counter() - started) / 4, 0.001)
                    if not ok or frame is None:
                        continue
                    if nominal_fps > 0:
                        capture_seconds = max(capture_seconds, 1 / nominal_fps)
                    if self._original_threads is None:
                        self._tune_threads(frame, benchmark)
                    benchmark(frame)
                    started = perf_counter()
                    for _ in range(5):
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            break
                        benchmark(frame)
                    elapsed = perf_counter() - started
                    if not ok or frame is None:
                        continue
                    height, width = frame.shape[:2]
                    modes.append(CameraMode(backend, mjpg, width * height, 1 / capture_seconds,
                                            min(5 / elapsed, 1 / capture_seconds)))
                finally:
                    capture.release()

        if modes:
            fastest = max(mode.total_fps for mode in modes)
            modes.sort(key=lambda mode: (
                mode.pixels if mode.total_fps >= 0.9 * fastest else 0,
                mode.total_fps,
            ), reverse=True)
        for mode in modes:
            capture = self._open(mode.backend, mode.mjpg)
            ok, frame = capture.read()
            if ok and frame is not None:
                self._capture = capture
                self._first_frame = frame
                height, width = frame.shape[:2]
                self.mode_info = (f"{capture.getBackendName()}, {width}×{height}, "
                                  f"потоков OpenCV: {cv2.getNumThreads()}, "
                                  f"захват {mode.capture_fps:.1f} кадр/с, "
                                  f"оценка с обработкой {mode.total_fps:.1f} кадр/с")
                return
            capture.release()
        self._capture = self._open(cv2.CAP_ANY, False)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._first_frame is not None:
            frame, self._first_frame = self._first_frame, None
            return True, frame
        if self._capture is None:
            return False, None
        return self._capture.read()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._original_threads is not None:
            cv2.setNumThreads(self._original_threads)
            self._original_threads = None

    @property
    def name(self) -> str:
        return f"camera:{self.index}"


class ScreenSource(FrameSource):
    is_live = True
    supports_background = True

    def __init__(self, region: Optional[dict]):
        self._sct = mss()
        self._region = region if region is not None else self._sct.monitors[0]

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        screenshot = self._sct.grab(self._region)
        frame = np.asarray(screenshot)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return True, frame

    def release(self) -> None:
        self._sct.close()

    @property
    def name(self) -> str:
        return "screen"


def parse_region(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("region must be left,top,width,height")
    left, top, width, height = parts
    return {"left": left, "top": top, "width": width, "height": height}


def create_source(
    source: str, input_path: Optional[str], camera: int, region: Optional[dict],
    camera_benchmark: Optional[Callable[[np.ndarray], None]] = None,
) -> FrameSource:
    if source == "image":
        if not input_path:
            raise ValueError("--input is required for image source")
        return ImageSource(input_path)
    if source == "video":
        if not input_path:
            raise ValueError("--input is required for video source")
        return VideoSource(input_path)
    if source == "camera":
        return CameraSource(camera, camera_benchmark)
    if source == "screen":
        return ScreenSource(region)
    raise ValueError(f"unknown source: {source}")
