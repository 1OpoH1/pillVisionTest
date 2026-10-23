from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from splitting import Instance


@dataclass
class ScaleInfo:
    count: int
    median_area: float
    median_size: float


def estimate_scale(instances: List[Instance]) -> Optional[ScaleInfo]:
    if not instances:
        return None
    areas = np.array([instance.area for instance in instances], dtype=np.float64)
    median_area = float(np.median(areas))
    return ScaleInfo(
        count=len(instances),
        median_area=median_area,
        median_size=2.0 * float(np.sqrt(median_area / np.pi)),
    )


def pixels_per_mm(reference_pixels: float, reference_mm: float) -> float:
    if reference_mm <= 0:
        raise ValueError("reference_mm must be positive")
    return reference_pixels / reference_mm


def sizes_mm(instances: List[Instance], px_per_mm: float) -> List[float]:
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    return [instance.size / px_per_mm for instance in instances]
