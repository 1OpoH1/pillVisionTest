from __future__ import annotations

from typing import Optional, Tuple

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


class CameraSource(FrameSource):
    is_live = True
    supports_background = True

    def __init__(self, index: int):
        self.index = index
        self._capture = cv2.VideoCapture(index)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._capture.read()
        return bool(ok), frame

    def release(self) -> None:
        self._capture.release()

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


def create_source(source: str, input_path: Optional[str], camera: int, region: Optional[dict]) -> FrameSource:
    if source == "image":
        if not input_path:
            raise ValueError("--input is required for image source")
        return ImageSource(input_path)
    if source == "video":
        if not input_path:
            raise ValueError("--input is required for video source")
        return VideoSource(input_path)
    if source == "camera":
        return CameraSource(camera)
    if source == "screen":
        return ScreenSource(region)
    raise ValueError(f"unknown source: {source}")
