from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from config import Config
from contact import Contact
from splitting import Instance, compute_gradient


def _color_for_id(identifier: int) -> tuple[int, int, int]:
    hue = (identifier * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _put_text(image: np.ndarray, text: str, position: tuple[int, int], scale: float = 0.6, color=(255, 255, 255)) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_result(
    frame: np.ndarray,
    instances: List[Instance],
    contacts: List[Contact],
    extra: Optional[str] = None,
) -> np.ndarray:
    canvas = frame.copy()
    for instance in instances:
        color = _color_for_id(instance.id)
        cv2.drawContours(canvas, [instance.contour], -1, color, 2)
        center = (int(round(instance.centroid[0])), int(round(instance.centroid[1])))
        cv2.circle(canvas, center, 3, color, -1)
        label = f"{instance.id}"
        if instance.confidence < 0.5:
            label += "?"
        _put_text(canvas, label, (center[0] + 5, center[1] - 5), 0.55, color)

    for contact in contacts:
        point_a = (int(round(contact.point_a[0])), int(round(contact.point_a[1])))
        point_b = (int(round(contact.point_b[0])), int(round(contact.point_b[1])))
        cv2.line(canvas, point_a, point_b, (0, 0, 255), 2, cv2.LINE_AA)
        midpoint = ((point_a[0] + point_b[0]) // 2, (point_a[1] + point_b[1]) // 2)
        cv2.circle(canvas, midpoint, 4, (0, 0, 255), -1)

    text = f"Objects: {len(instances)}   Contacts: {len(contacts)}"
    if extra:
        text += f"   {extra}"
    _put_text(canvas, text, (12, 26), 0.7, (255, 255, 255))
    return canvas


def make_debug_views(frame: np.ndarray, mask: np.ndarray, config: Config) -> Dict[str, np.ndarray]:
    mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    if maximum > 0:
        distance_view = np.clip(distance / maximum * 255.0, 0, 255).astype(np.uint8)
    else:
        distance_view = np.zeros_like(mask)
    distance_color = cv2.applyColorMap(distance_view, cv2.COLORMAP_JET)
    gradient = compute_gradient(frame, config)
    gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    overlay = frame.copy()
    overlay[mask > 0] = (0, 200, 0)
    foreground = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return {
        "Mask": mask_view,
        "Distance": distance_color,
        "Gradient": gradient_color,
        "Foreground": foreground,
    }
