from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config import Config


class Segmenter:
    def __init__(self, config: Config):
        self.config = config
        self._background_image: Optional[np.ndarray] = None
        self._background_arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def set_background(self, image: np.ndarray) -> None:
        if image is None:
            return
        self._background_image = image.copy()
        self._background_arrays = None

    def clear_background(self) -> None:
        self._background_image = None
        self._background_arrays = None

    @property
    def has_background(self) -> bool:
        return self._background_image is not None

    @staticmethod
    def _to_bgr(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    @staticmethod
    def _chroma(image: np.ndarray) -> np.ndarray:
        data = image.astype(np.float32)
        total = data.sum(axis=2, keepdims=True) + 1e-6
        return data / total * 255.0

    @staticmethod
    def _estimate_background_color(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        border = max(2, min(h, w) // 60)
        parts = [
            frame[:border, :, :],
            frame[-border:, :, :],
            frame[:, :border, :],
            frame[:, -border:, :],
        ]
        pixels = np.concatenate([part.reshape(-1, 3) for part in parts], axis=0)
        return np.median(pixels, axis=0)

    def _prepare_background(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._background_image is None:
            color = self._estimate_background_color(frame)
            image = np.empty_like(frame)
            image[:, :] = np.clip(color, 0, 255).astype(np.uint8)
        else:
            image = self._background_image
            if image.shape[:2] != frame.shape[:2]:
                image = cv2.resize(image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = self._chroma(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return lab, chroma, gray

    def _get_background(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._background_arrays is None or self._background_image is None:
            self._background_arrays = self._prepare_background(frame)
        return self._background_arrays

    def _threshold(self, dist: np.ndarray) -> float:
        if self._background_image is not None or not self.config.auto_color_threshold:
            return float(self.config.color_threshold)
        dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(np.clip(otsu, self.config.color_threshold_min, self.config.color_threshold_max))

    def foreground(self, frame: np.ndarray) -> np.ndarray:
        frame = self._to_bgr(frame)
        background_lab, background_chroma, background_gray = self._get_background(frame)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        dist = np.linalg.norm(frame_lab - background_lab, axis=2)
        threshold = self._threshold(dist)
        mask = dist > threshold

        if self.config.shadow_suppress:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frame_chroma = self._chroma(frame)
            chroma_diff = np.linalg.norm(frame_chroma - background_chroma, axis=2)
            shadow = (frame_gray < background_gray - 2.0) & (chroma_diff < self.config.chroma_threshold)
            mask &= ~shadow

        mask_u8 = mask.astype(np.uint8) * 255
        if self.config.open_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.open_kernel, self.config.open_kernel))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        if self.config.close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.close_kernel, self.config.close_kernel))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        if self.config.fill_holes:
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(mask_u8)
            if contours:
                cv2.drawContours(filled, contours, -1, 255, -1)
            mask_u8 = filled

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        clean = np.zeros_like(mask_u8)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= self.config.min_area:
                clean[labels == index] = 255
        return clean
